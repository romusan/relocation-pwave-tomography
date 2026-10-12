"""Meaningful migration gates: geometry, association, native rays and solvers."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from regional.geometry import Projection, Grid, grids, candidate_nodes
from regional.inputs import associate, active_stations, spatial_order
from regional.forward import FieldCache
from regional.rays import trace, kernels
from regional.inversion import regularized, station_design
from regional.q_inversion import covariance_whitener, run_q
from regional.pipeline import relocate
from regional.downloads import download
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def config():
    d = json.loads((ROOT/'configuracion_regional.json').read_text(encoding='utf-8'))
    d['ray_backend'] = 'cpp'
    return d


class Migration(unittest.TestCase):
    def setUp(self):
        self.domain = {'lat_min':4.,'lat_max':6.,'lon_min':-77.,'lon_max':-75.,'prof_min':0.,'prof_max':180.}
        self.proj = Projection(self.domain)
        self.cfg = config()

    def test_projection_roundtrip_and_geodesic_scale(self):
        audit = self.proj.audit(self.domain)
        self.assertLess(audit['roundtrip_max_deg'], 1e-9)
        self.assertLess(audit['pair_distance_max_relative_error_percent'], .1)

    def test_separate_grids_and_depth_not_limited_to_vmm(self):
        direct, coarse = grids(self.domain, self.proj,10.,20.)
        self.assertEqual(coarse.spacing,2*direct.spacing)
        center = direct.nodes([0,0,10.])
        cand = candidate_nodes(direct,center,10,self.domain,self.proj)
        self.assertIn(0.,direct.xyz(cand)[:,2])
        center = direct.nodes([0,0,170.])
        cand = candidate_nodes(direct,center,10,self.domain,self.proj)
        self.assertIn(180.,direct.xyz(cand)[:,2])
        with self.assertRaises(ValueError):
            grids(self.domain,self.proj,7.,20.)

    def test_station_epochs(self):
        table = pd.DataFrame({'net':['CM','CM'],'sta':['A','A'],'lat':[5.,5.1],'lon':[-76.,-76.],
            'inicio':['2000-01-01','2020-01-01'],'fin':['2019-12-31','']})
        self.assertEqual(active_stations(table,'2018-01-01').lat.iloc[0],5.)
        self.assertEqual(active_stations(table,'2022-01-01').lat.iloc[0],5.1)

    def test_association_preserves_raw_and_traceable_rejections(self):
        stations = pd.DataFrame({'net':['CM']*4,'sta':['A','B','C','D'],'lat':[5.1,4.9,5.,5.],
            'lon':[-76.,-76.,-76.1,-75.9],'inicio':['2000-01-01']*4,'fin':['']*4})
        rows=[]
        t0=pd.Timestamp('2022-01-01T00:00:00Z')
        for st in stations.itertuples():
            x,y=self.proj.xy(st.lat,st.lon)
            t=np.sqrt(float(x*x+y*y)+20**2)/6.
            rows.append({'event_id':'E','origin_time':str(t0),'ev_lat':5.,'ev_lon':-76.,'ev_prof_km':20.,
                'net':st.net,'sta':st.sta,'fase':'P','pick_time':str(t0+pd.Timedelta(seconds=t)),'prob':.9})
        rows += [dict(rows[0]), dict(rows[1],pick_time=str(t0-pd.Timedelta(seconds=1))),
                 dict(rows[2],pick_time=str(t0+pd.Timedelta(seconds=80)))]
        raw=pd.DataFrame(rows)
        original=raw.copy(deep=True)
        retained,audit,events=associate(raw,stations,self.domain,self.proj,(np.array([0.,180.]),np.array([6.,6.])),self.cfg)
        pd.testing.assert_frame_equal(raw,original)
        self.assertEqual(len(audit),len(raw))
        self.assertEqual(len(retained),4)
        self.assertEqual(len(events),1)
        self.assertEqual(events.n_p_stations.iloc[0],4)
        self.assertIn('duplicate',set(audit.reason))
        self.assertIn('nonpositive_travel_time',set(audit.reason))
        self.assertIn('outside_association_window',set(audit.reason))

    def test_native_rays_match_reference_and_length_on_axis(self):
        k,j,i=np.indices((11,11,11))
        field=np.ascontiguousarray(np.sqrt(k*k+j*j+i*i)/6.)
        finish=0
        for node in [(10,0,0),(10,7,4),(1,1,1)]:
            start=np.ravel_multi_index(node,field.shape)
            ref=trace(field,start,finish,1.,'reference')
            native=trace(field,start,finish,1.,'cpp')
            np.testing.assert_array_equal(ref[0],native[0])
            np.testing.assert_allclose(ref[1],native[1],rtol=1e-15)
        self.assertEqual(trace(field,1210,0,1.,'cpp')[1].sum(),10.)
        self.assertIsNone(trace(np.ones((5,5,5)),124,0,1.,'cpp'))

    def test_native_rays_with_fsm_fields(self):
        grid=Grid(np.arange(0.,55.,5.),np.arange(0.,55.,5.),np.arange(0.,55.,5.))
        station=pd.DataFrame({'station':['CM.A'],'st_x_km':[0.],'st_y_km':[0.]})
        with tempfile.TemporaryDirectory() as tmp:
            cache=FieldCache(np.full(grid.shape,6.),grid,station,tmp,self.cfg)
            field=cache.get('CM.A')
            a=trace(field,grid.linear(np.array([[8,6,3]]))[0],0,5.,'cpp')
            b=trace(field,grid.linear(np.array([[8,6,3]]))[0],0,5.,'reference')
            np.testing.assert_array_equal(a[0],b[0])
            np.testing.assert_allclose(a[1],b[1])
            reloaded=FieldCache(np.full(grid.shape,6.),grid,station,tmp,self.cfg)
            np.testing.assert_array_equal(reloaded.get('CM.A'),field)
            self.assertEqual(reloaded.loaded,1)
            changed=FieldCache(np.full(grid.shape,6.1),grid,station,tmp,self.cfg)
            self.assertNotEqual(cache.model_key,changed.model_key)
            cache.close()
            reloaded.close()
            changed.close()

    def test_gauge_and_lsmr_recovery(self):
        cfg=dict(self.cfg,laplace_weight=0.,cell_damping=1e-8,station_damping=1e-8)
        A=sparse.csr_matrix([[1.,0.],[0.,1.],[1.,1.]])
        truth=np.array([.01,-.02])
        x,_,info=regularized(A,A@truth,np.ones((1,1,2),bool),sparse.csr_matrix((3,0)),np.ones(3),cfg)
        np.testing.assert_allclose(x,truth,atol=1e-10)
        self.assertTrue(info['converged'])
        S,free,reference=station_design(pd.DataFrame({'station':['B','A','C']}))
        self.assertEqual(reference,'A')
        self.assertEqual(free,['B','C'])
        np.testing.assert_array_equal(S.toarray()[1],np.zeros(2))

    def test_covariance_whitening(self):
        cov=np.array([[4.,1.],[1.,9.]])
        W=covariance_whitener(cov,np.array([0,1]))
        np.testing.assert_allclose(W@cov@W.T,np.eye(2),atol=1e-12)

    def test_short_spatial_prefix_spans_domain(self):
        lat,lon=np.meshgrid(np.linspace(4.1,5.9,10),np.linspace(-76.9,-75.1,10))
        events=pd.DataFrame({'event_id':[f'E{i}' for i in range(lat.size)],
            'lat':lat.ravel(),'lon':lon.ravel(),'depth_km':np.full(lat.size,40.),'gap_deg':np.full(lat.size,100.)})
        order=spatial_order(events,self.proj,cell_km=10.)
        selected=events.set_index('event_id').loc[order[:10]]
        self.assertGreater(selected.lon.max()-selected.lon.min(),1.5)
        self.assertGreater(selected.lat.max()-selected.lat.min(),1.5)
        self.assertEqual(order,spatial_order(events,self.proj,cell_km=10.))

    def test_origin_correction_is_absolute_across_rounds(self):
        grid=Grid(np.arange(-20.,25.,5.),np.arange(-20.,25.,5.),np.arange(0.,45.,5.))
        stations=pd.DataFrame({'station':['A','B','C','D'],'st_x_km':[-15.,15.,0.,0.],'st_y_km':[0.,0.,-15.,15.]})
        with tempfile.TemporaryDirectory() as tmp:
            cache=FieldCache(np.full(grid.shape,6.),grid,stations,tmp,self.cfg)
            node=tuple(grid.nodes([0.,0.,20.]))
            observations=np.array([cache.get(s)[node]+2. for s in stations.station])
            picks=pd.DataFrame({'event_id':['E']*4,'station':stations.station,'observed_s':observations,
                'st_lat_current':[5.]*4,'st_lon_current':[-76.]*4})
            cfg=dict(self.cfg,relocation_radius_km=0.)
            n,o,_=relocate(cache,picks,{'E':node},{'E':0.},{},self.domain,self.proj,cfg)
            n2,o2,_=relocate(cache,picks,n,o,{},self.domain,self.proj,cfg)
            self.assertEqual(o['E'],2.)
            self.assertEqual(o2['E'],2.)
            self.assertEqual(n2,n)
            cache.close()

    def test_download_transient_retry_and_window_identity(self):
        temporary=Mock(status_code=503,content=b'Unavailable')
        valid=Mock(status_code=200,content=b'x'*3000)
        session=Mock()
        session.get.side_effect=[temporary,valid]
        with tempfile.TemporaryDirectory() as tmp:
            params={'starttime':'2022-01-01','endtime':'2022-01-02'}
            path,status=download(session,'https://example.invalid',params,tmp,retries=2,pause=0.)
            self.assertEqual(status['status'],'downloaded')
            self.assertEqual(len(status['attempts']),2)
            self.assertEqual(path.stat().st_size,3000)
            cached,_=download(session,'https://example.invalid',params,tmp,retries=1)
            self.assertEqual(session.get.call_count,2)
            self.assertEqual(cached,path)
            session.get.side_effect=[Mock(status_code=204,content=b'')]
            no_data,status=download(session,'https://example.invalid',dict(params,endtime='2022-01-03'),tmp)
            self.assertIsNone(no_data)
            self.assertEqual(status['status'],'no_data')

    def test_download_failure_not_permanent_empty_marker(self):
        session=Mock()
        session.get.return_value=Mock(status_code=503,content=b'fail')
        with tempfile.TemporaryDirectory() as tmp:
            _,status=download(session,'https://example.invalid',{},tmp,retries=1)
            self.assertEqual(status['status'],'temporary_error')
            self.assertFalse(list(Path(tmp).glob('*.mseed')))
            session.get.return_value=Mock(status_code=200,content=b'x'*3000)
            path,status=download(session,'https://example.invalid',{},tmp,retries=1)
            self.assertIsNotNone(path)
            self.assertEqual(status['status'],'downloaded')

    def test_q_constant_model_recovers_zero_perturbation(self):
        grid=Grid(np.arange(-20.,25.,5.),np.arange(-20.,25.,5.),np.arange(0.,45.,5.))
        stations=pd.DataFrame({'station':['A','B','C','D'],'st_x_km':[-15.,15.,0.,0.],'st_y_km':[0.,0.,-15.,15.]})
        cfg=dict(self.cfg,min_ray_hits=1)
        with tempfile.TemporaryDirectory() as tmp:
            cache=FieldCache(np.full(grid.shape,6.),grid,stations,Path(tmp)/'cache',cfg)
            nodes={f'E{n}':tuple(grid.nodes([5.*(n%2),0.,15.+5*(n//2)])) for n in range(6)}
            endpoint=pd.DataFrame([(e,s) for e in nodes for s in stations.station],columns=['event_id','station'])
            E,_=kernels(cache,endpoint,nodes,grid,'cpp',q_mode=True)
            pairs=[]
            for n,e in enumerate(nodes):
                for a,b in ((0,1),(2,3)):
                    obs=float((E[n*4+a].sum()-E[n*4+b].sum())/cfg['q_initial'])
                    pairs.append({'event_id':e,'station_i':stations.station.iloc[a],
                        'station_j':stations.station.iloc[b],'delta_tstar_s':obs,'sigma_delta_tstar_s':.01,'qc_ok':True})
            csv=Path(tmp)/'pairs.csv'
            pd.DataFrame(pairs).to_csv(csv,index=False)
            split=pd.DataFrame({'event_id':list(nodes),'split':['train']*5+['holdout']})
            summary=run_q(csv,pd.DataFrame(),split,cache,nodes,grid,self.domain,self.proj,cfg,tmp)
            self.assertLess(summary['train_rms_after_s'],1e-12)
            self.assertLess(summary['holdout_rms_after_s'],1e-12)
            self.assertEqual(summary['retained_pairs'],12)
            cache.close()


if __name__=='__main__':
    unittest.main()

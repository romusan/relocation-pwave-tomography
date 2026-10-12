import unittest
import pandas as pd
from seleccion_zona import extend_cohort
from regional.geometry import Projection


class Selection(unittest.TestCase):
    def test_base_split_preserved_and_only_exact_departments_or_coast_added(self):
        base=pd.DataFrame(dict(event_id=['a','b'],n_p_stations=[4,4],lat=[8,7],lon=[-74,-75],
            depth_km=[100,120],gap_deg=[100,110],split=['train','holdout']))
        other=pd.DataFrame(dict(event_id=['c','d','e','f'],n_p_stations=[4]*4,lat=[3,4,3.8,8],
            lon=[-76,-76,-77.3,-75],depth_km=[80,40,20,100],gap_deg=[90,100,80,90]))
        available=pd.concat([base.drop(columns='split'),other])
        catalog=pd.DataFrame(dict(event_id=['a','b','c','d','e','f'],region=['Colombia','Colombia',
            'Caldono - Cauca, Colombia','Buga - Valle del Cauca, Colombia','Océano Pacífico',
            'Caucasia - Antioquia, Colombia']))
        projection=Projection(dict(lat_min=2,lat_max=9,lon_min=-79,lon_max=-73))
        result,extra,eligible=extend_cohort(base,available,catalog,projection,3)
        self.assertEqual(result.iloc[:2].split.tolist(),['train','holdout'])
        self.assertEqual(set(extra.event_id),{'c','d','e'})
        self.assertEqual(set(eligible.event_id),{'c','d'})
        self.assertEqual(extra.split.eq('holdout').sum(),1)
        self.assertFalse(result.event_id.duplicated().any())
        with self.assertRaises(ValueError):
            extend_cohort(base,available,catalog,projection,5)


if __name__=='__main__': unittest.main()

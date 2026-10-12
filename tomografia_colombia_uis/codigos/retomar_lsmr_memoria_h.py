"""Use the verified RAM field cache and the unchanged LSMR recovery driver."""
from pathlib import Path
import json,os,hashlib
import paralelizar_jacobiano_h as parallel
import cache_memoria_h
h=parallel.h
cache_memoria_h.enable(h)
h.save(h.OUT/'memory_cache_execution.json',dict(pid=os.getpid(),fields_in_memory_mb=2048,new_disk_fields=False,existing_disk_fields_read=True,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
driver=Path(__file__).with_name('retomar_lsmr_h.py')
exec(compile(driver.read_text(encoding='utf-8'),str(driver),'exec'),{'__file__':str(driver),'__name__':'__main__'})

"""Debug script to see what's being extracted from competitors"""
import sys
sys.path.insert(0, 'G:\\Shared drives\\Dialfa\\INFO PRECIOS PARA EL SR. DIEGO F')

from data_extractors.cintolo_extractor import CintoloExtractor
from data_extractors.zaloze_extractor import ZalozeExtractor

print("=" * 80)
print("CINTOLO DATA:")
print("=" * 80)
cintolo = CintoloExtractor()
cintolo_data = cintolo.extract()
print(cintolo_data)

print("\n" + "=" * 80)
print("ZALOZE DATA:")
print("=" * 80)
zaloze = ZalozeExtractor()
zaloze_data = zaloze.extract()
print(zaloze_data)



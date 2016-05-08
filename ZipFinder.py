import requests
from time import sleep
from geopy.geocoders import Nominatim
import geopy.exc as geoexception

def BuildAdress(name, number):
    return "{} {}, Detroit, MI".format(number, name)


def BuildAdressComplete(name, number):
    return "{} {}, Detroit, Michigan".format(number, name)


class GoogleZipFinder:
    def __init__(self):
        self._count=0
        
    def getzip(self, adress):
        if self._count>= 2500:
            return None
        try:
            sleep(0.1)
            info = { "address":adress, "key":"AIzaSyAjY-LZF2VSYx6q54mHzLKI48d9dytp388"}
            self._count += 1
            r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=info)
            jsonresp = r.json()
            comp = jsonresp["results"][0]["address_components"]
            for el in comp:
                if "postal_code" in el["types"]:
                    return int(el["long_name"])
            return None
        except:
            return None
        

class NominatimZipFinder:
    def __init__(self):
        self.ok = True
    
    def GetInfo(self, buildname):
        if not self.ok:
            print('service error')
            return None
        try:
            sleep(2)           
            geolocator = Nominatim(timeout=2)
            location1 = geolocator.geocode(buildname, timeout=3)
            return location1.address.replace(", United States of America","")
        except (geoexception.GeocoderQuotaExceeded):
            self.ok = False
            print('service error')
            return None
        except Exception as error:
            print(buildname, " => ", error)
            return None
        
    def GetZipcode(self, name):
        try:
            return int(name[-5:])
        except:
            return None
        
    def getzip(self, adress):
        try:
            return self.GetZipcode(self.GetInfo(adress))
        except:
            return None
        
 
if __name__ == '__main__':
    z = NominatimZipFinder()
    print(z.getzip("19221 Runyon Street, Detroit, Michigan"))
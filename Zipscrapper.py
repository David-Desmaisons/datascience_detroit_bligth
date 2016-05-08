from bs4 import BeautifulSoup
import requests
import re

def stripcontent(element):
    return element.contents[0].strip()


def formatnumber(value):
    return float(value.strip().replace("$","").replace(",","").replace("%","").replace(" years","").replace(" people",""))


class ZipScrapper:
    def __init__(self):
        self._relevant = {"Persons Per Householde:", "Average House Value:", "Income Per Household:", "Median Age:",
                          "Population:", "White Population:", "Black Population:", "Female Population:", "Male Population:"}


    def parseZip(self, zipnumber):
        r = requests.get("http://zipcode.org/{}".format(zipnumber))
        soup = BeautifulSoup(r.text, 'html.parser')
        res = {}
        for text in [f for f in soup.find_all("div", class_='Zip_Code_HTML_Block_Label') if stripcontent(f) in self._relevant]:
            key = stripcontent(text)
            key = key.replace(" ","").replace(":","")
            value = stripcontent(text.find_next_siblings(class_='Zip_Code_HTML_Block_Text')[0])
            res[key] = formatnumber(value)
        return res


class ZipScrapper2:
    def __init__(self):
        self._key ={"Estimated zip code population in 2013:": "Population2013",
                    "Zip code population in 2010:": "Population2010",
                    "Zip code population in 2000": "Population2000",
                    "Houses and condos" : "HouseholdNumber",
                    "Males": "MalePopulation",
                    "Females": "FemalePopulation",
                    "Unemployed": "UnemployedPercentage",
                    "Estimated median house ": "MedianHouseValue"}
        self._list ={"White population": "WhitePopulation",
                     "Black population": "BlackPopulation"}
        self._table ={"Median resident age:": "MedianAge",
                      "Average household size:": "PersonsPerHousehold",
                      "Estimated median household income in 2013:": "IncomePerHousehold",
                      "Percentage of family households:": "PercentageFamilyHouseholds",
                      "Residents with income below the poverty level in 2013:": "PercentageBelowPoverty",
                      "Residents with income below 50% of the poverty level in 2013": "PercentageBelow50Poverty",
                      "Renter-occupied apartments" : "PercentageRenters"}


    def parseZip(self, zipnumber):
        r = requests.get("http://www.city-data.com/zips/{}.html".format(zipnumber))
        soup = BeautifulSoup(r.text, 'html.parser')
        res = {}
        for name, namekey in self._key.items():
            element = soup.find("b", text = re.compile(name))
            res[namekey] = formatnumber(element.next_sibling)
        for name, namekey in self._table.items():
            element = soup.find("b", text = re.compile(name))
            nexttable = element.find_next("table")
            p = nexttable.find("p", class_="h").next_sibling
            res[namekey] = formatnumber(p)
        print(res)
        tabletitle = soup.find("h3",text = re.compile("Races in zip code"))
        table = tabletitle.parent.parent
        listelement = table.find("li", class_="col-md-7")
        for individualelement in listelement.find_all("li"):
            if individualelement.contents[1] in self._list:
                value = formatnumber(individualelement.contents[0].contents[0])
                res[self._list[individualelement.contents[1]]] = value
        return res


if __name__ == '__main__':
    z = ZipScrapper2()
    print(z.parseZip(48201))
    z = ZipScrapper()
    print(z.parseZip(48201))


class tranfer_market_api:
    def __init__(self):
        self.urls = []
        for year in range(2005, 2019):
            urls.append("https://www.transfermarkt.com/"
                                "premier-league/startseite/wettbewerb/GB1/plus/?saison_id={0}".format(year))


    
    def print(self):
        print self.urls
        
if __name__ == "__main__":
    
    
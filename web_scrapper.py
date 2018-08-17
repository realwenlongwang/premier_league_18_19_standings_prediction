import requests
from bs4 import BeautifulSoup
import pandas as pd


class web_scrapper:
    def __init__(self):
        self.__headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                        'Chrome/47.0.2526.106 Safari/537.36'}
        self.__urls = []
        self.__years = range(2005, 2019)
        for year in self.__years:
            self.__urls.append(
                "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/"
                "GB1/plus/?saison_id={0}".format(year))

        self.__responses = {}
        self.dfs = {}

    def test_print(self):
        for url in self.__urls:
            print(url)

    def connect(self):
        # Request the HTMLs
        for url, year in zip(self.__urls, self.__years):
            response = requests.get(url, headers=self.__headers)
            # Connect failed
            if response.status_code != 200:
                raise requests.ConnectionError("Request Failed!")
            self.__responses[year] = response
        print("Request Succeeded!")

    def digest(self):
        for year, response in self.__responses.iteritems():
            # Create the summary DataFrame for each year and store it to dictionary
            self.dfs[year] = self._cook(response)
        return self.dfs

    def _cook(self, response):
        # ================ scrap the summary table ===========================
        soup = BeautifulSoup(response.content, "lxml")
        # find the two tables
        tables = soup.find_all("div", {"class": "responsive-table"})

        # Scrap the clubs full name and short name from the table
        clubs = tables[0].find_all("a", {"class": "vereinprofil_tooltip"})

        club_full_name_list = []
        club_short_name_list = []
        for i in range(len(clubs)):
            if i % 3 == 1:
                club_full_name_list.append(clubs[i].text)
            elif i % 3 == 2:
                club_short_name_list.append(clubs[i].text)

        summary_dict = {"Club Full Name": club_full_name_list, "Club Short Name": club_short_name_list}

        # Scrap the market values
        values = soup.find_all("td", {"class": "rechts hide-for-small hide-for-pad"})
        total_mv_list = []
        avg_mv_list = []
        for i in range(len(values)):
            # Skip the first row becuase it is the sum up mv
            if i % 2 == 0 and i != 0:
                total_mv_list.append(values[i].text)
            elif i % 2 == 1 and i != 1:
                avg_mv_list.append(values[i].text)

        summary_dict["Total Market Values"] = total_mv_list
        summary_dict["Avg. Market Values"] = avg_mv_list

        ages = soup.find_all("td", {"class": "zentriert hide-for-small hide-for-pad"})

        avg_player_age_list = []
        # Skip the first row
        for i in range(1, len(ages)):
            avg_player_age_list.append(ages[i].text)

        summary_dict["Avg. Player age"] = avg_player_age_list
        # Convert the table to DataFrame
        summary_df = pd.DataFrame(data=summary_dict)

        # ================ scrap the standings table ===========================
        standings = tables[1].find_all("td", {"class": "zentriert"})
        clubs_standing = tables[1].find_all("a", {"class": "vereinprofil_tooltip"})

        club_standing_name_list = []

        for i in range(len(clubs_standing)):
            if i % 2 == 1:
                club_standing_name_list.append(clubs_standing[i].text)

        goal_difference_list = []
        points_list = []
        for i in range(len(standings)):
            # +/- column on the website
            if i % 4 == 2:
                goal_difference_list.append(standings[i].text)
            # Pts column
            elif i % 4 == 3:
                points_list.append(standings[i].text)
        standing_dict = {"Club Short Name": club_standing_name_list, "Goal Difference": goal_difference_list,
                         "Points": points_list}
        # Convert the standing table to the DataFrame
        standing_df = pd.DataFrame(data=standing_dict)

        # Create the ranking column
        standing_df = standing_df.reset_index()
        standing_df["index"] += 1
        standing_df = standing_df.rename(columns={"index": "Position"})

        # Merge two DataFrames to a big table
        summary_df = pd.merge(summary_df, standing_df, on="Club Short Name")

        return summary_df


if __name__ == "__main__":
    my_web_scrapper = web_scrapper()
    try:
        my_web_scrapper.connect()
    except requests.ConnectionError as detail:
        print(detail)
    my_web_scrapper.digest()
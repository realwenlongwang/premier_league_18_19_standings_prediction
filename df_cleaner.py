import pandas as pd


def numeric_data(string):
    string = string.replace(",", ".")
    tokens = string.split(" ")
    # This is age column
    if len(tokens) == 1:
        tokens[0] = float(tokens[0])
    else:  # This is the market values column
        if tokens[1] == "Mill.":
            tokens[0] = float(tokens[0])
        elif tokens[1] == "Bill.":
            tokens[0] = float(tokens[0]) * 1000
            # print(tokens[0])
        elif tokens[1] == "Th.":
            tokens[0] = float(tokens[0]) / 1000.0
            # print(tokens[0])
    return tokens[0]


class df_cleaner:

    def __init__(self):
        self.__numeric_df = pd.DataFrame()

    def cook(self, raw_df):
        self.__numeric_df = raw_df.copy()
        self.__change_columns_name(raw_df=raw_df)
        self.__numeric_df.iloc[:, [0, 1, 4]] = self.__numeric_df.iloc[:, [0, 1, 4]].applymap(numeric_data)
        self.__numeric_df[["Goal Difference", "Points"]] = self.__numeric_df[["Goal Difference", "Points"]].applymap(lambda x: int(x))
        return self.__numeric_df

    def __change_columns_name(self, raw_df):
        unit_string = raw_df.iloc[0, 0].split(" ")[1:]
        unit_string = " ".join(unit_string)
        # Don't use df.column.values, will cause issue
        self.__numeric_df.rename(columns={"Avg. Market Values": "Avg. Market Values" + "(" + unit_string + ")",
                                          "Total Market Values": "Total Market Values" + "(" + unit_string + ")"},
                                 inplace=True
                                 )


if __name__ == "__main__":
    my_df_cleaner = df_cleaner()
    my_test_df = pd.read_pickle("./obj/big_summary_df.pkl")
    my_pretty_df = my_df_cleaner.cook(my_test_df)
    print my_pretty_df.info()

import sys


def load_data():
    # format data : [user id, item id / movie id, rating, timestamp]
    datas = []
    file_data = open("data/u.data", "r")
    for data in file_data:

        # rating = [int(rate) for rate in data.split()]
        rating = data.split()

        datas.append(rating)

    file_data.close()
    return datas


def load_item():
    # movie id | movie title | release date | video release date | IMDb URL | genre.....
    # NOTE : 1682 DATA

    if sys.version_info < (3, 0):
        # python 2
        file_item = open("data/u.item", "r")
    else:
        # python 3
        # in version 3, python will encode all string default in utf-8 hiks hiks,
        # so if u want to load data with another encoding, please write it explicitly
        file_item = open("data/u.item", "r", encoding="ISO-8859-1")

    items = []
    for data in file_item:
        # data tabbed | separated values
        items.append(data[:-1].split("|"))
    file_item.close()
    return items


def load_info():

    file_info = open("data/u.info", "r")

    info = dict()
    for i, data in enumerate(file_info):

        if i == 0:
            info["users"] = int(data.split()[0])

        elif i == 1:
            info["items"] = int(data.split()[0])

        elif i == 2:
            info["ratings"] = int(data.split()[0])

    # {'items': 1682, 'users': 943, 'ratings': 100000}
    return info

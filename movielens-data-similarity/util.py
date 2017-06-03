from collections import OrderedDict
from math import sqrt

from file_handler import load_data, load_info, load_item

movies = load_item()
ratings = load_data()


def find_movie_by_title(title):
    index_title = 1
    title_right_index = -7 # index trim
    for movie in movies:
        # check movie title by trim it from right first
        if movie[index_title][:title_right_index] == title:
            return movie

    return None


def get_movie_id_by_title(title):
    movie = find_movie_by_title(title)
    index_movie_id = 0
    if movie is not None:
        return movie[index_movie_id]
        # return int(movie[index_movie_id])
    else:
        return None


def get_jaccard_coefficient(first_movie_id, second_movie_id):

    # list for check user already have rated
    user_rate_list = []
    double_rate = 0

    index_user = 0
    index_movie = 1

    for rate in ratings:

        movie_selected_id = rate[index_movie]

        if movie_selected_id == first_movie_id or movie_selected_id == second_movie_id:

            user_id = rate[index_user]

            if user_id not in user_rate_list:
                user_rate_list.append(user_id)
            else:
                double_rate += 1

    # jaccard coefficient = user rate both movies / user rated at least one movies
    return round(float(double_rate) / len(user_rate_list),3)


def get_correlation(first_movie_id, second_movie_id):

    # list for check user already have rated
    single_rated_user = []

    first_movie_rating = OrderedDict()
    second_movie_rating = OrderedDict()

    index_user = 0
    index_movie = 1
    index_rating = 2

    # get first and second movie which user bot have rated those movies
    for rate in ratings:

        movie_selected_id = rate[index_movie]

        if movie_selected_id == first_movie_id or movie_selected_id == second_movie_id:

            user_id = rate[index_user]

            # First insert all rating
            if movie_selected_id == first_movie_id:
                first_movie_rating[user_id] = int(rate[index_rating])

            elif movie_selected_id == second_movie_id:
                second_movie_rating[user_id] = int(rate[index_rating])

            # Get Single Voted User
            if user_id not in single_rated_user:
                single_rated_user.append(user_id)
            else:
                # remove user from single rated if user voted both movie
                single_rated_user.remove(user_id)

    # remove rating single voted
    for user_id in single_rated_user:

        if user_id in first_movie_rating:
            del first_movie_rating[user_id]
        else:
            del second_movie_rating[user_id]

    first_rating_list = []
    for user_id_key in sorted(first_movie_rating):
        # first_rating_list.append((user_id_key, first_movie_rating[user_id_key]))
        first_rating_list.append(first_movie_rating[user_id_key])

    second_rating_list = []
    for user_id_key in sorted(first_movie_rating):
        #  second_rating_list.append((user_id_key, second_movie_rating[user_id_key]))
        second_rating_list.append(second_movie_rating[user_id_key])

    print(first_rating_list)
    print(second_rating_list)
    return calculate_correlation(first_rating_list,second_rating_list)


def calculate_correlation(x,y):

    # get vector dimension
    dim = len(x)
    # print("n", dim)
    # return 0 when the number of users who have rated both is so low that
    if len(x) < 5:
        return 0
    else:

        mean_x = float(sum(x)) / len(x)
        mean_y = float(sum(y)) / len(y)

        top = 0
        for i in range(dim):
            top += (x[i] - mean_x) * (y[i] - mean_y)

        left, right = 0, 0
        for i in range(dim):
            left += (x[i] - mean_x) ** 2
            right += (y[i] - mean_y) ** 2

        down = sqrt(left) * sqrt(right)

        try:
            correlation = top / down
        except ZeroDivisionError:
            correlation = 0

        return round(correlation,4)

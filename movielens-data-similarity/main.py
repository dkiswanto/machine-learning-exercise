from file_handler import load_data, load_info,load_item
from util import *
import time

def main():

    # Load Data
    movies = load_item()

    # Question 6.a.i
    print("Question 6.a.i")
    toy_story_id = get_movie_id_by_title("Toy Story")
    golden_eye_id = get_movie_id_by_title("GoldenEye")
    print("Jaccard coefficient Toy Story and Golden Eye",
          get_jaccard_coefficient(toy_story_id, golden_eye_id))
    
    # Question 6.a.ii
    print("\nQuestion 6.a.ii")
    three_colors_red_id = get_movie_id_by_title("Three Colors: Red")
    three_colors_blue_id = get_movie_id_by_title("Three Colors: Blue")
    print("Jaccard coefficient Three Colors: Red & Blue",
          get_jaccard_coefficient(three_colors_red_id, three_colors_blue_id))
    
    # Question 6.a.iii
    print("\nQuestion 6.a.iii")
    movie_name = "Taxi Driver"
    taxi_driver_id = get_movie_id_by_title(movie_name)
    temp = []
    print("Searching best five jaccard coef, @", movie_name, "....")
    for movie in movies:
        # Check its not same movie
        if (taxi_driver_id != movie[0]):
            coefficient = get_jaccard_coefficient(taxi_driver_id, movie[0])
            data = [coefficient, movie[1]]
            temp.append(data)
    
    best_five = sorted(temp, reverse=True)[:5]
    for i in best_five:
        print(i)
    
    # Question 6.a.iiii
    print("\nQuestion 6.a.iv - Favorite Movie")
    fav_movie = "Shawshank Redemption, The"
    fav_movie_id = get_movie_id_by_title(fav_movie)
    temp = []
    print("Searching best five jaccard coef, @", fav_movie, "....")
    for movie in movies:
        # Check if its not same movie
        if (fav_movie_id != movie[0]):
            coefficient = get_jaccard_coefficient(fav_movie_id, movie[0])
    
            data = [coefficient, movie[1]]
            temp.append(data)
    
    best_five = sorted(temp, reverse=True)[:5]
    
    for movie in best_five:
        print(movie)
    
    # Question 6.b.i
    print("\nQuestion 6.b.i", '- function correlation located in util.py')
    mov_1_title, mov_2_title = "Back to the Future", "Shawshank Redemption, The"
    mov_1 = get_movie_id_by_title(mov_1_title)
    mov_2 = get_movie_id_by_title(mov_2_title)
    print("Correlation", mov_1_title, "and" , mov_2_title,
        get_correlation(mov_1, mov_2))
    
    # Question 6.b.ii
    print("\nQuestion 6.b.ii")
    mov_1_title, mov_2_title = "Toy Story", "GoldenEye"
    mov_1 = get_movie_id_by_title(mov_1_title)
    mov_2 = get_movie_id_by_title(mov_2_title)
    print("Correlation", mov_1_title, "and" , mov_2_title,
        get_correlation(mov_1, mov_2))
    
    # Question 6.b.iii
    print("\nQuestion 6.b.iii")
    mov_1_title, mov_2_title = "Taxi Driver", "Lost in Space"
    mov_1 = get_movie_id_by_title(mov_1_title)
    mov_2 = get_movie_id_by_title(mov_2_title)
    print("Correlation", mov_1_title, "and" , mov_2_title,
        get_correlation(mov_1, mov_2))

    # Question 6.b.iv
    print("\nQuestion 6.b.iv")
    movie_title = "Star Wars"
    movie_id = get_movie_id_by_title(movie_title)
    temp = []
    print("Searching best five correlation, @", movie_title, "....")
    for movie in movies:
        # Check if its not same movie
        if (movie_id != movie[0]):
            correlation = get_correlation(movie_id, movie[0])
    
            data = [correlation, movie[1]]
            temp.append(data)
    
    best_five = sorted(temp, reverse=True)[:5]
    for movie in best_five:
        print(movie)
    
    # Question 6.b.v
    print("\nQuestion 6.b.v - Favorite Movie")
    movie_title = "Shawshank Redemption, The"
    movie_id = get_movie_id_by_title(movie_title)
    temp = []
    print("Searching best five correlation, @", movie_title, "....")
    for movie in movies:
        # Check if its not same movie
        if (movie_id != movie[0]):
            correlation = get_correlation(movie_id, movie[0])
    
            data = [correlation, movie[1]]
            temp.append(data)
    
    best_five = sorted(temp, reverse=True)[:5]
    for movie in best_five:
        print(movie)
    


if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (round(time.time() - start_time, 2)))



import tmdbsimple as tmdb

import urllib
import os
from IPython.display import display, HTML, Image



key_v4 = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxMGY0MGYwZDVkNzk0ZTRiYWNiMjY2MTg4MTI4YTg5NiIsInN1YiI6IjViZGE1NjNlMGUwYTI2MDNjYTAwM2Q1MCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.6yPX2IdoGMMDQ_yjXkj9CyIFG0c6c6qcOaxYn7hC_RQ'
key_v3 = '10f40f0d5d794e4bacb266188128a896'

tmdb_connector = tmdb
tmdb_connector.API_KEY = key_v3

# https://api.themoviedb.org/3/movie/2/images/7VqciAWfFkYrFK7XlQXVjej1Fup.jpg?api_key=10f40f0d5d794e4bacb266188128a896
# http://image.tmdb.org/t/p/w185//
"""
w92", "w154", "w185", "w342", "w500", "w780" is the size of image in the url
"""


download_path = "/Users/zishuoli/Doc/project/Feature_extractor/Image/tmdb_movie_poster/"


# review_path = "/Users/zishuoli/Doc/project/Feature_extractor/Image/Review/"
# def review_download(id,path_download,path_download=review_path,key=key_v3):
#     #https://developers.themoviedb.org/3/movies/get-movie-reviews
#     url = "https://api.themoviedb.org/3/movie/{}/reviews?api_key={}&language=en-US&page=1".format(id,key)
#     content = request.get(url)
#     return content

def tmdb_img_download(id, path_download=download_path ,tmdbPath='http://image.tmdb.org/t/p/w185/'):
    """
    Func:
        Donwload data from TMBD website based on movie Id
    Input:
        id: TMDB movie id
        path_db
    """
    mov = tmdb.Movies(id)
    info = mov.info()
    title = info['title']

    url_img = tmdbPath + info['poster_path']

    folderPath = os.path.join(path_download,str(id))
    try:
        os.mkdir(folderPath)
    except:
        pass

    path_download = os.path.join(path_download,str(id),title+".jpg")
    try:
        urllib.request.urlretrieve(url_img, path_download)
        print(f"Success: Poster for Film {id} is successfully downloaded")
    except:
        print("Fail: Film {}'s poster is not available".format(id))
        os.remove(folderPath)




def display_images(id, path_db='http://image.tmdb.org/t/p/w185/'):

    mov = tmdb.Movies(id)
    info = mov.info()

    img_path = info['poster_path']

    path = path_db + img_path
    images = ''

    images += "<img style='width: 100px; margin: 0px; \
            float: left; border: 1px solid black;' src='%s' />" \
            % path

    display(HTML(images))




def cosine_matrix(data):
    '''
    Row based similirty
    
    default: user is in row, item in column
    ''' 
    dim = data.shape
    
    constant = tf.constant(1e-9,dtype=tf.float32)

    df = tf.placeholder(shape=[dim[0],dim[1]],dtype=tf.float32)

    similar_user = tf.matmul(df,tf.transpose(df)) + constant

    norm_user = tf.reshape(tf.sqrt(tf.diag_part(similar_user)),[-1,1])

    norm_user_matrix = tf.matmul(norm_user,tf.transpose(norm_user))

    similar_user = similar_user/norm_user_matrix


    with tf.Session() as sess:
        ans = sess.run(similar_user,feed_dict={df:data})

    return ans
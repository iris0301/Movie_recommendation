class recommend(object):
    def svd_recommend(rating_matrix,movieId_vs_tmdbId,top_k_products=10):
        nb_users = rating_matrix.shape[0]
        nb_products = rating_matrix.shape[1]

        # Create a Tensorflow graph
        graph = tf.Graph()

        with graph.as_default():
            user_item_matrix = tf.placeholder(tf.float32, shape=(nb_users, nb_products))

            # SVD
            S, U, V = tf.svd(user_item_matrix)

            S = tf.diag(S)        
        
            Su = tf.matmul(U, tf.sqrt(S))
            Si = tf.matmul(tf.sqrt(S), tf.transpose(V))

            # Compute user ratings
            ratings_t = tf.matmul(Su, Si)
            
            # Pick top k suggestions
            best_ratings_t, best_items_t = tf.nn.top_k(ratings_t, top_k_products)

        # Create Tensorflow session
        session = tf.InteractiveSession(graph=graph)

        # Compute the top k suggestions for all users
        

        best_items,best_items_rating = session.run([best_items_t,best_ratings_t],feed_dict = {
            user_item_matrix: rating_matrix
        })

        # convert to tmdb id
        best_items = np.array(best_items)
        
        ans_items = []
        for row in best_items:
            tmp = []
            for item in row:
                try:
                    tmp.append(movieId_vs_tmdbId[str(item)])
                except:
                    continue

            ans_items.append(tmp)
        # return item are TMDB ID and corresponding predictive rating
        return ans_items,best_items_rating


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
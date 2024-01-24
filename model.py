import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout


class MovieRecommender:
    def __init__(self, num_users, num_movies, embedding_size=10):
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.model = self._build_model()

    def _build_model(self):
        # User Input
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(self.num_users + 1, self.embedding_size, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)

        # Movie Input
        movie_input = Input(shape=(1,), name='movie_input')
        movie_embedding = Embedding(self.num_movies + 1, self.embedding_size, name='movie_embedding')(movie_input)
        movie_vec = Flatten(name='movie_flatten')(movie_embedding)

        # Concatenate Features
        concat = Concatenate()([user_vec, movie_vec])

        # Fully Connected Layers
        fc1 = Dense(128, activation='relu')(concat)
        fc1_dropout = Dropout(0.2)(fc1)
        fc2 = Dense(64, activation='relu')(fc1_dropout)
        fc2_dropout = Dropout(0.2)(fc2)
        fc3 = Dense(32, activation='relu')(fc2_dropout)

        # Output Layer
        output = Dense(1)(fc3)

        # Create Model
        model = Model([user_input, movie_input], output)
        return model

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, train_user_ids, train_movie_ids, train_ratings, epochs=10, batch_size=32):
        self.model.fit([train_user_ids, train_movie_ids], train_ratings, epochs=epochs, batch_size=batch_size)

    def predict(self, user_ids, movie_ids):
        return self.model.predict([user_ids, movie_ids])
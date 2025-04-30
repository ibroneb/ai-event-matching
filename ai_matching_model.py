# ai_matching_model.py

import tensorflow as tf
import pandas as pd

# Sample users and events
user_ids = ['user_1', 'user_2', 'user_3']
event_ids = ['event_101', 'event_102', 'event_103']

# Simulated interaction data
interactions = pd.DataFrame({
    'user_id': ['user_1', 'user_1', 'user_2', 'user_3'],
    'event_id': ['event_101', 'event_102', 'event_102', 'event_103']
})

# Create vocab layers
user_vocab = tf.keras.layers.StringLookup(vocabulary=user_ids)
event_vocab = tf.keras.layers.StringLookup(vocabulary=event_ids)

# Convert to tensor indices
user_indices = user_vocab(interactions['user_id'])
event_indices = event_vocab(interactions['event_id'])

# Neural matching model
class MatchingModel(tf.keras.Model):
    def __init__(self, num_users, num_events, embedding_dim=16):
        super().__init__()
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.Embedding(num_users + 1, embedding_dim),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])
        self.event_embedding = tf.keras.Sequential([
            tf.keras.layers.Embedding(num_events + 1, embedding_dim),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])
        self.dot_product = tf.keras.layers.Dot(axes=1)

    def call(self, user_input, event_input):
        user_vec = self.user_embedding(user_input)
        event_vec = self.event_embedding(event_input)
        return self.dot_product([user_vec, event_vec])

# Instantiate and compile model
model = MatchingModel(len(user_ids), len(event_ids))
model.compile(optimizer='adam', loss='mse')

# Labels: 1 = positive interaction
labels = tf.ones((len(user_indices), 1))

# Train model
model.fit([user_indices, event_indices], labels, epochs=10, verbose=1)

# Predict for new user-event pair
test_user = user_vocab(['user_2'])
test_event = event_vocab(['event_103'])
score = model.predict([test_user, test_event])
print(f"Predicted match score: {score[0][0]:.4f}")

"""Recomendador colaborativo de productos con KNeighborsClassifier."""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class ProductRecommender:
	"""Recomienda productos a partir de puntuaciones usuario-producto."""

	def __init__(self, n_neighbors=5, min_rating=4.0):
		self.n_neighbors = n_neighbors
		self.min_rating = min_rating
		self.model = None

	def fit(self, ratings, user_col="user_id", product_col="product_id", rating_col="rating"):
		required = {user_col, product_col, rating_col}
		if not required.issubset(ratings.columns):
			raise ValueError(f"El DataFrame debe incluir las columnas: {sorted(required)}")
		if ratings.empty:
			raise ValueError("El conjunto de puntuaciones está vacío.")

		self.user_col, self.product_col = user_col, product_col
		self.users = pd.Index(ratings[user_col].unique())
		self.products = pd.Index(ratings[product_col].unique())
		self.user_to_idx = {user: i for i, user in enumerate(self.users)}
		self.product_to_idx = {product: i for i, product in enumerate(self.products)}
		self.matrix = (
			ratings.pivot_table(index=user_col, columns=product_col, values=rating_col, aggfunc="mean")
			.reindex(index=self.users, columns=self.products)
			.fillna(0.0)
		)

		profiles = self.matrix.to_numpy(dtype=float)
		n_products = len(self.products)
		features, labels = [], []
		for user, product, rating in ratings[[user_col, product_col, rating_col]].itertuples(index=False, name=None):
			user_idx = self.user_to_idx[user]
			product_idx = self.product_to_idx[product]
			profile = profiles[user_idx].copy()
			profile[product_idx] = 0.0  # Evita revelar la puntuación que se clasifica.
			product_vector = np.zeros(n_products, dtype=float)
			product_vector[product_idx] = 1.0
			features.append(np.concatenate((profile, product_vector)))
			labels.append(int(float(rating) >= self.min_rating))

		self.model = KNeighborsClassifier(
			n_neighbors=min(max(1, int(self.n_neighbors)), len(features)),
			weights="distance",
		)
		self.model.fit(np.asarray(features), np.asarray(labels))
		return self

	def recommend(self, user_id, n=5):
		"""Devuelve una lista de (producto, probabilidad estimada de gustar)."""
		if self.model is None:
			raise ValueError("Llama a fit() antes de solicitar recomendaciones.")
		if user_id not in self.user_to_idx:
			raise ValueError(f"Usuario desconocido: {user_id}")
		if n <= 0:
			return []

		profile = self.matrix.iloc[self.user_to_idx[user_id]].to_numpy(dtype=float)
		candidates = np.flatnonzero(profile == 0.0)
		if candidates.size == 0:
			return []

		samples = []
		for product_idx in candidates:
			product_vector = np.zeros(len(self.products), dtype=float)
			product_vector[product_idx] = 1.0
			samples.append(np.concatenate((profile, product_vector)))

		probabilities = self.model.predict_proba(np.asarray(samples))
		positive = np.flatnonzero(self.model.classes_ == 1)
		scores = probabilities[:, positive[0]] if positive.size else np.zeros(len(candidates))
		ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
		return [(self.products[index], float(score)) for index, score in ranked[:n]]


if __name__ == "__main__":
	ratings = pd.DataFrame(
		[
			("Ana", "Libro", 5), ("Ana", "Café", 4), ("Ana", "Auriculares", 2),
			("Luis", "Libro", 4), ("Luis", "Café", 5), ("Luis", "Planta", 3),
			("Marta", "Auriculares", 5), ("Marta", "Planta", 4), ("Marta", "Libro", 2),
			("Pablo", "Café", 2), ("Pablo", "Auriculares", 4), ("Pablo", "Planta", 5),
		],
		columns=["user_id", "product_id", "rating"],
	)
	recommender = ProductRecommender(n_neighbors=3, min_rating=4).fit(ratings)
	print("Recomendaciones para Ana:", recommender.recommend("Ana", n=3))

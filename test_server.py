from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_text_simila():
    response = client.post(
        "/text_simila",
        headers={'accept': 'application/json','Content-Type': 'application/json'},
        json={
            "corpus": [
                "私は春です","私は秋です","僕は春夏秋冬です"
            ],
            "morpheme_on":True
        },
    )
    assert response.status_code == 200
    assert response.json() == {
  "cos_similarity": [
    [
      1,
      1,
      0.5085423203783267
    ],
    [
      1,
      1,
      0.5085423203783267
    ],
    [
      0.5085423203783267,
      0.5085423203783267,
      1
    ]
  ],
  "tfidf": [
    [
      1,
      0
    ],
    [
      1,
      0
    ],
    [
      0.5085423203783267,
      0.8610369959439764
    ]
  ],
  "feature_name": [
    "です",
    "春夏秋冬"
  ]
}
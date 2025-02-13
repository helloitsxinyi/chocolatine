import os

name = "test"
apiurl = "https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw/asn/"
kafkaconf = {
    'modellertopic': 'model-requests',
    'bootstrap-model': ['localhost:9092'],
    'group': 'model-consumer-group'
}

dbconf = {
    'name': 'models',
    'host': 'localhost',
    'port': 5432,
    'password': os.getenv('DB_PASSWORD')
}


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
def blight_model():
    traind = pd.read_csv('train.csv', encoding='latin1')
    testd = pd.read_csv('test.csv')
    traind.dropna(subset=['compliance'], inplace=True)
    traind = traind[['compliance', 'ticket_id', 'violation_street_number',  'judgment_amount','fine_amount', 'admin_fee','state_fee','late_fee']]
    testd = testd[['ticket_id', 'violation_street_number',  'judgment_amount','fine_amount', 'admin_fee','state_fee','late_fee']]
    y = traind['compliance']
    x = traind.drop('compliance',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    kn = LogisticRegression()
    kn.fit(X_train, y_train)
    t = kn.predict_proba(testd)
    return pd.Series(t[:,1], index=testd['ticket_id'])

blight_model()


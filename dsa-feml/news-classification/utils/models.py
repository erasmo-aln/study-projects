from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB


def voting_classifier():

    model_1 = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=30, max_iter=1000)
    model_2 = MultinomialNB()
    model_3 = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=1)

    model_list = [('lg', model_1), ('nb', model_2), ('rf', model_3)]

    voting_model = VotingClassifier(estimators=model_list, voting='soft')

    return voting_model


def stacking_classifier():

    model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_2 = MultinomialNB()
    model_3 = LogisticRegression(multi_class='multinomial', random_state=30, max_iter=1000)

    model_list = [('rf', model_1), ('nb', model_2)]

    stacking_model = StackingClassifier(estimators=model_list, final_estimator=model_3)

    return stacking_model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from PIL import Image
import streamlit as st
import io
import plotly.express as px
import plotly.graph_objects as go

Img = Image.open("Vitamin.png")

def example(content):
    st.markdown(f'<p style="text-align:center;background-image: linear-gradient(to right,#32A852, #2402F4);color:#FFFFFF;font-size:30px;font-family:Georgia Bold;border-radius:2%;">{content}</p>', unsafe_allow_html=True)

def example2(content):
    st.markdown(f'<p style="text-align:left;font-size:22px;font-family:Georgia Bold;border-radius:2%;">{ content }</p>', unsafe_allow_html=True)

def example3(content):
    st.markdown(f'<p style="text-align:left;font-size:18px;font-family:Georgia Bold;border-radius:2%;">{ content }</p>', unsafe_allow_html=True)

def example4(content):
    st.markdown(f'<p style="text-align:left;font-size:14px;font-family:Georgia Bold;border-radius:2%;">{ content }</p>', unsafe_allow_html=True)

def example5(content):
    st.markdown(f'<p style="text-align:center;background-image: linear-gradient(to right, #2402F4,#32A852);color:#FFFFFF;font-size:22px;font-family:Georgia Bold;border-radius:2%;">{content}</p>', unsafe_allow_html=True)

pdata = pd.read_csv('plasmadata.csv')

Accdf = pd.DataFrame()
Accdf['Classifier'] = ["K-Nearest Neighbors", "Decision Tree", "Gaussian Naive Bayes","Random Forest"]

pdata['BETAPLASMALEVEL']=0
pdata['RETPLASMALEVEL']=0

for i in range(len(pdata['BETAPLASMA'])):
    k = pdata['BETAPLASMA'][i]
    if(k <= 300 and k >= 50):
        pdata['BETAPLASMALEVEL'][i] = 1
    else:
        pdata['BETAPLASMALEVEL'][i] = 0

for J in range(len(pdata['RETPLASMA'])):
    m = pdata['RETPLASMA'][J]/10
    if(m >= 20.1 and m <= 80.2 ):
        pdata['RETPLASMALEVEL'][J] = 1
    else:
        pdata['RETPLASMALEVEL'][J] = 0

pdata['SEX'] = pdata['SEX'].astype('category')
pdata['SMOKSTAT'] = pdata['SMOKSTAT'].astype('category')
pdata['VITUSE'] = pdata['VITUSE'].astype('category')
pdata['BETAPLASMALEVEL'] = pdata['BETAPLASMALEVEL'].astype('category')
pdata['RETPLASMALEVEL'] = pdata['RETPLASMALEVEL'].astype('category')

Afig = px.imshow(round(pdata.corr(),2),color_continuous_scale='algae', origin='lower',text_auto=True)

def BetaPlasmaData(pdata):
    X1 = pdata.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
    y1 = pdata.iloc[:,14]
    return(X1,y1)

def RetPlasmaData(pdata):
    X1 = pdata.iloc[:,[0,1,2,3,4,5,6,7,8,9,11]]
    y1 = pdata.iloc[:,15]
    return(X1,y1)

def TrainTest(X1,y1):
    return train_test_split(X1, y1,test_size=0.3,random_state=3)      

def scaling(X_train, X_test, Scaler):
    Scaler.fit(X_train)
    return(Scaler.transform(X_train),Scaler.transform(X_test))
        
def KNN(X_train, X_test, y_train, y_test): 
    Acc_Sc,confusion_matrix,my_predictions = classifier(neighbors.KNeighborsClassifier(n_neighbors = 3),X_train, X_test, y_train, y_test)
    return(Acc_Sc,confusion_matrix,my_predictions)

def DTC(X_train, X_test, y_train, y_test):
    Acc_Sc,confusion_matrix,my_predictions = classifier(DecisionTreeClassifier(criterion = 'gini', random_state = 0),X_train, X_test, y_train, y_test)
    return(Acc_Sc,confusion_matrix,my_predictions)

def GNB(X_train, X_test, y_train, y_test):
    Acc_Sc,confusion_matrix,my_predictions = classifier(GaussianNB(),X_train, X_test, y_train, y_test)
    return(Acc_Sc,confusion_matrix,my_predictions)

def RFC(X_train, X_test, y_train, y_test):
    Acc_Sc,confusion_matrix,my_predictions = classifier(RandomForestClassifier(random_state=0),X_train, X_test, y_train, y_test)
    return(Acc_Sc,confusion_matrix,my_predictions)

def classifier(my_classifier,X_train, X_test, y_train, y_test):
    my_classifier.fit(X_train, y_train)
    my_predictions = my_classifier.predict(X_test)
    Acc_Sc = accuracy_score(y_test, my_predictions)
    confusion_matrix = metrics.confusion_matrix(y_test, my_predictions)
    return(Acc_Sc,confusion_matrix,my_predictions)


st.title(
    """
    ## Dietary Pattern-Related Plasma Levels of Retinol and Beta-Carotene.
    """
)


with st.sidebar:
    tab1, tab2, tab3 = st.tabs(["Introduction", "Data Analysis", "Machine Learning"])
    with tab1:
        Ar = st.radio('',("Description","Data set Information","Variables of the Dataset"))

    with tab2:
        Br = st.radio('',("Initial Data Analysis","Data Visualization"))

    with tab3:
        Cr = st.radio('',("Regression Analysis","Scikit-learn Estimators","Feature Engineering","Cross Validation","Hyperparameter Tuning","Prediction"))
    
    example4("")
    example4("")
    example4("")
    example4("")
    example4("")
    example4("")
    example4("")
    example4("")
    example4("")
    example4("")
    example3("Author:  Harish Neelam")
    example4("Email:   harineelam10@gmail.com")

if(Ar == "Description"):
    st.image(Img)
    example('Introduction')
    example2(" Description:  ")
    st.write("""
- Some Research studies has revealed that low plasma concentrations of retinol, beta-carotene, or other carotenoids may be linked to a higher risk of developing certain forms of cancer.
- However, only a small number of research have looked into the factors that affect the plasma concentrations of these micronutrients.
- I wish to create a Machine learning model which determines the association between dietary components, personal traits, and plasma concentrations of retinol and beta-carotene.
- Study subjects were patients who underwent elective surgery to biopsy or remove a lesion of the lung, colon, breast, skin, ovary, or uterus within a three-year period. Only two analytes are present in the data.
""")

if(Ar == "Data set Information"):
    example('Introduction')
    example2("Data set Information:")
    with st.expander("Data set Information:"):
        st.write(""" 
        - This Data set is obtained from https://hbiostat.org/data/repo/plasma.html.
        - Reference: Nierenberg DW, Stukel TA, Baron JA, Dain BJ, Greenberg ER. Determinants of plasma levels of beta-carotene and retinol. American Journal of Epidemiology 1989;130:511-521.
        - This Data contains 315 observations on 16 variables.
        """)
    st.dataframe(pdata)

if(Ar == "Variables of the Dataset"):
    example('Introduction')
    example2("Variables of the Dataset:")
    st.markdown("""
    
    Variables Names:
    
    - AGE: Age (years).
	- SEX: Sex (1=Male, 2=Female).
	- SMOKSTAT: Smoking status (1=Never, 2=Former, 3=Current Smoker).
	- QUETELET: Quetelet (weight/(height^2)).
	- VITUSE: Vitamin Use (1=Yes, fairly often, 2=Yes, not often, 3=No).
	- CALORIES: Number of calories consumed per day.
	- FAT: Grams of fat consumed per day.
	- FIBER: Grams of fiber consumed per day.
	- ALCOHOL: Number of alcoholic drinks consumed per week.
	- CHOLESTEROL: Cholesterol consumed (mg per day).
	- BETADIET: Dietary beta-carotene consumed (mcg per day).
	- RETDIET: Dietary retinol consumed (mcg per day).
	- BETAPLASMA: Plasma beta-carotene (ng/ml).
	- RETPLASMA: Plasma Retinol (ng/ml).
    - BETAPLASMALEVEL: Levels of Beta plasma (1=Normal,0=Abnormal).
    - RETPLASMALEVEL: Levels of Beta plasma (1=Normal,0=Abnormal).
    """)

example('Data Analysis')
if(Br == "Initial Data Analysis"):
    example2("Initial Data Analysis:")
    with st.expander("Info"):
        buffer = io.StringIO()
        pdata.info(buf=buffer)
        pinf = buffer.getvalue()
        st.text(pinf)
    with st.expander('Describe'):
        st.write(pdata.describe())
    with st.expander("Correlation"):
        Atb, Btb = st.tabs(['Matrix','Heatmap'])
        with Atb:
            st.write(pdata.corr())
            st.write("""
            - BETAPLASMA values are strongly correlated with AGE, QUETELET, FIBER, CHOLESTROL and BETADIET.
            - RETPLASMA values are strongly correlated with AGE, FAT, ALCOHOL and RETDIET.
            """)
        with Btb:
            st.plotly_chart(Afig)
            st.write("""
            - BETAPLASMA values are strongly correlated with AGE, QUETELET, FIBER, CHOLESTROL and BETADIET.
            - RETPLASMA values are strongly correlated with AGE, FAT, ALCOHOL and RETDIET.
            """)

if(Br == "Data Visualization"):
    example2("Data Visualization:")
    st.write("Let's see some plots before we go further.")
    BrAs = st.selectbox(
        'Variable on X-axis:',
        ('AGE', 'SEX', 'SMOKSTAT', 'QUETELET', 'VITUSE', 'CALORIES', 'FAT',
       'FIBER', 'ALCOHOL', 'CHOLESTEROL', 'BETADIET', 'RETDIET', 'BETAPLASMA',
       'RETPLASMA'))
    BrBs = st.selectbox(
        'Variable on Y-axis:',
        ('QUETELET', 'SEX', 'SMOKSTAT', 'AGE', 'VITUSE', 'CALORIES', 'FAT',
       'FIBER', 'ALCOHOL', 'CHOLESTEROL', 'BETADIET', 'RETDIET', 'BETAPLASMA',
       'RETPLASMA'))
    BrAr = st.radio(
        'Do you want to see the categorized plots?',
        ('No','Yes')
    )
    if(BrAr == 'No'):
        Bfig = px.scatter(data_frame=pdata, x= BrAs,y =BrBs,color_discrete_sequence=['darkgreen','darkblue','limegreen'])
    else:
        BrCs = st.selectbox(
        'Variable as hue:',
        ('SEX', 'SMOKSTAT', 'VITUSE','BETAPLASMALEVEL','RETPLASMALEVEL'))
        Bfig = px.scatter(data_frame=pdata, x= BrAs,y =BrBs,color=BrCs,color_discrete_sequence=['darkgreen','darkblue','limegreen'])
    Bfig.update_traces(marker=dict(size=10))
    st.plotly_chart(Bfig)

def stAccCM(Acc_Sc,confusion_matrix,clr):
    example3("Accuracy:")
    st.write(Acc_Sc)
    example3("Confusion Matrix:")
    st.plotly_chart(px.imshow(confusion_matrix,color_continuous_scale=clr, origin='lower',text_auto=True))
    

def ClassTabs(X_train, X_test, y_train, y_test):
    CrAAtab1, CrAAtab2, CrAAtab3,CrAAtab4 = st.tabs(["K-Nearest Neighbors", "Decision Tree", "Gaussian Naive Bayes","Random Forest"])
    with CrAAtab1:
        Acc_Sc,confusion_matrix,my_predictions = KNN(X_train, X_test, y_train, y_test)
        stAccCM(Acc_Sc,confusion_matrix,'ice')
    with CrAAtab2:
        Acc_Sc,confusion_matrix,my_predictions = DTC(X_train, X_test, y_train, y_test)
        stAccCM(Acc_Sc,confusion_matrix,'algae')
    with CrAAtab3:
        Acc_Sc,confusion_matrix,my_predictions = GNB(X_train, X_test, y_train, y_test)
        stAccCM(Acc_Sc,confusion_matrix,'haline')
    with CrAAtab4:
        Acc_Sc,confusion_matrix,my_predictions = RFC(X_train, X_test, y_train, y_test)
        stAccCM(Acc_Sc,confusion_matrix,'Emrld')

def yd(x,w):
    k = w[0][0]
    for i in range(len(w)):
        k = k + w[i][0]*x**i
    return k

example('Machine Learning')
if(Cr == "Regression Analysis"):
    example2("Regression Analysis:")
    example4("There are many variables in data, as features are increased the model fits exactly to the data. The goal is to extract the model which fits best to the data not the most. The below regression analysis will help in figuring out which regression model fit the data best.")
    CrSb1 = st.selectbox("Select the Target:",('BETAPLASMA','RETPLASMA'))
    CrMs1 = st.multiselect("Select atleast a Feature:",['AGE', 'QUETELET', 'CALORIES', 'FAT', 'FIBER', 'ALCOHOL', 'CHOLESTEROL', 'BETADIET', 'RETDIET'],['AGE'])
    x = np.array([pdata[CrMs1[0]]]).T
    ones = np.ones((x.shape[0],1))
    for i in range(1,len(CrMs1)):
        x = np.append(x,np.array([pdata[CrMs1[i]]]).T,axis = 1)
    X = np.append(ones,x,axis =1)
    y = np.array([pdata[CrSb1]]).T
    w = np.linalg.pinv(X.T @ X)@ X.T @ y
    xd = np.linspace(-400,400,100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xd, y=yd(xd,w),mode='lines',name='Regression line',line = dict(color='darkblue',width = 3)))
    fig.add_trace(go.Scatter(y=pdata[CrSb1],mode='markers',name='Original Data',marker = dict(color='limegreen',size = 8,symbol="star")))
    st.plotly_chart(fig)
        




if(Cr == 'Scikit-learn Estimators'):
    example2("Classifiers in Scikit-learn:")
    example4("There are many classifier estimators in scikit-learn and some of them gave best accuracies for the data. They are as follows:")
    CrAA = st.selectbox('',("Beta Plasma Classification","Retinol Plasma Classification"))
    if(CrAA == "Beta Plasma Classification"):
        X1, y1 = BetaPlasmaData(pdata)
        X_train, X_test, y_train, y_test = TrainTest(X1,y1)
        ClassTabs(X_train, X_test, y_train, y_test)

    else:
        X1, y1 = RetPlasmaData(pdata)
        X_train, X_test, y_train, y_test = TrainTest(X1,y1)
        ClassTabs(X_train, X_test, y_train, y_test)

def Accs(X_tr,X_te,y_tr,y_te):
    Accs_class = [KNN(X_tr,X_te,y_tr,y_te)[0],DTC(X_tr,X_te,y_tr,y_te)[0],
                  GNB(X_tr,X_te,y_tr,y_te)[0],RFC(X_tr,X_te,y_tr,y_te)[0]]
    return Accs_class

def Accs_preds(X_tr,X_te,y_tr,y_te):
    Accs_class = [KNN(X_tr,X_te,y_tr,y_te)[2],DTC(X_tr,X_te,y_tr,y_te)[2],
                  GNB(X_tr,X_te,y_tr,y_te)[2],RFC(X_tr,X_te,y_tr,y_te)[2]]
    return Accs_class

def Acc_plot(Acc_1,Acc_2):
    Accdf['Initial'] = Acc_1
    Accdf['Scaled'] = Acc_2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = Accdf['Classifier'],y = Accdf['Initial'],mode='lines+markers',name = 'Initial',line = dict(color='darkblue')))
    fig.add_trace(go.Scatter(x = Accdf['Classifier'],y = Accdf['Scaled'],mode='lines+markers', name = 'Scaled',line = dict(color='green')))
    fig.update_layout(width = 400, height = 500)
    st.plotly_chart(fig)

def PcaTabs(Acc_1,Acc_preds1,CrNC,ind):
    Sclr = Normalizer()
    X1_scaled = Sclr.fit_transform(X1)
    pca = PCA(n_components = CrNC)
    pca.fit(X1_scaled)
    X_pca = pca.transform(X1_scaled)
    X_train, X_test, y_train, y_test = TrainTest(X_pca,y1)
    Acc_1 = Accs(X_train, X_test, y_train, y_test)
    Acc_preds1 = Accs_preds(X_train, X_test, y_train, y_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test,mode='markers',name='Actual',marker=dict(color='blue',size = 10,symbol = 'star')))
    fig.add_trace(go.Scatter( y=Acc_preds1[ind],mode='markers',name='Predicted',marker=dict(color='green',size = 5)))
    fig.update_layout(width = 700, height = 300)
    example4("Accuracy:")
    st.write(Acc_1[ind])
    example4("Actual vs Predicted:")
    st.plotly_chart(fig)
    


if(Cr == 'Feature Engineering'):
    example2("Feature Engineering:")
    CrBaTab1,CrBaTab2 = st.tabs(["Data Scaling","Feature Extraction"])
    with CrBaTab1:
        example3("Data Scaling:")
        example4("- Data can be scaled to train the model better. The same scaling method should be applied to test data to get the correct score. That can be done by many methods and two of those are shown below.")
        sfg = st.radio("",("Standardization","Normalization"))
        if(sfg == 'Standardization'):
            Scaler = StandardScaler()
        else:
            Scaler = Normalizer()
        
        Crcol1, Crcol2 = st.columns([2,1.7])
        with Crcol1:
            example3("Beta Plasma Classification:")
            example4("Comparision of Accuracies:")
            X1, y1 = BetaPlasmaData(pdata)
            X_train, X_test, y_train, y_test = TrainTest(X1,y1)
            X_train_scaled, X_test_scaled = scaling(X_train,X_test,Scaler)
            Acc_1 = Accs(X_train, X_test, y_train, y_test)
            Acc_2 = Accs(X_train_scaled, X_test_scaled, y_train, y_test)
            Acc_plot(Acc_1,Acc_2)
            
        with Crcol2:
            example3("Retinol Plasma Classification:")
            example4("Comparision of Accuracies:")
            X1, y1 = RetPlasmaData(pdata)
            X_train, X_test, y_train, y_test = TrainTest(X1,y1)
            X_train_scaled, X_test_scaled = scaling(X_train,X_test,Scaler)
            Acc_1 = Accs(X_train, X_test, y_train, y_test)
            Acc_2 = Accs(X_train_scaled, X_test_scaled, y_train, y_test)
            Acc_plot(Acc_1,Acc_2)
        

    with CrBaTab2:
        example3("Pricipal Component Analysis:")
        example4("- Here, Feature extraction is done by Principal component analysis. By changing the number of components, the features extracted are changed, which again changes the accuracy.")
        CrBs = st.selectbox('',("Beta Plasma Classification","Retinol Plasma Classification"))
        CrNC = st.slider("",1,11,1)
        if(CrBs == "Beta Plasma Classification"):
            X1, y1 = BetaPlasmaData(pdata)
        else:
            X1, y1 = RetPlasmaData(pdata)

        Crtab1, Crtab2, Crtab3, Crtab4 = st.tabs(["K-Nearest Neighbors", "Decision Tree", "Gaussian Naive Bayes","Random Forest"])
        with Crtab1:
            PcaTabs(X1,y1,CrNC,0)
        with Crtab2:
            PcaTabs(X1,y1,CrNC,1)
        with Crtab3:
            PcaTabs(X1,y1,CrNC,2)
        with Crtab4:
            PcaTabs(X1,y1,CrNC,3)

if(Cr == 'Cross Validation'):
    example2("Cross Validation:")
    example4("- In machine learning, we couldn't fit the model on the training data and can't say that the model will work accurately for the real data. For this, we must assure that our model got the correct patterns from the data, and it is not getting up too much noise. For this purpose, we use the cross-validation technique.") 
    example4("- Cross-validation is a technique used to assess a machine learning model and test its accuracy. It involves reserving a specific sample of a dataset on which the model isn't trained. Later on, the model is tested on this sample to evaluate it.")
    example4("- Here, I have used k-fold cross validation. This procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation.")
    CrBs = st.selectbox('',("Beta Plasma Classification","Retinol Plasma Classification"))
    if(CrBs == "Beta Plasma Classification"):
        X1, y1 = BetaPlasmaData(pdata)
    else:
        X1, y1 = RetPlasmaData(pdata)

    CrBsRa = st.radio("Choose a Scaling method:",("Standardization","Normalization"))
    CrBsRb = st.radio("Choose a Classifier:",("K-Nearest Neighbors", "Decision Tree", "Gaussian Naive Bayes","Random Forest"))
    CrBsRc = st.radio("Do you want to Include PCA?",("No","Yes"))
    CrBsRd = st.number_input("Enter no.of times to split data:",min_value = 1,max_value = 100,value=5)
    CrBsRe = st.radio("Select the type of metric:",("Accuracy score","F1 score"))
    
    if(CrBsRa == "Standardization"):
        Scaling_method = StandardScaler()
    else:
        Scaling_method = Normalizer()
    
    if(CrBsRc == "Yes"):
        CrNC = st.slider("",1,11,5)
        X1_scaled = Scaling_method.fit_transform(X1)
        pca = PCA(n_components = CrNC)
        pca.fit(X1_scaled)
        X1 = pca.transform(X1_scaled)
    
    if(CrBsRb == "K-Nearest Neighbors"):
        Classifier_method = neighbors.KNeighborsClassifier(n_neighbors = 3)
    elif(CrBsRb == "Decision Tree"):
        Classifier_method = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
    elif(CrBsRb == "Gaussian Naive Bayes"):
        Classifier_method = GaussianNB()
    else:
        Classifier_method = RandomForestClassifier(random_state=0)

    clf = make_pipeline(Scaling_method,Classifier_method)
    cv = ShuffleSplit(n_splits=CrBsRd, test_size=0.3, random_state=0)

    if(CrBsRe == "Accuracy score"):
        scores = cross_val_score(clf, X1, y1, cv=cv)
        example3("Average Accuracy Score:")
        example4("It means that the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.")
        st.write(np.mean(scores))
    else:
        scores = cross_val_score(clf, X1, y1, cv=cv,scoring = 'f1_weighted')
        example3("Average F1 Score:")
        example4("The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.")
        st.write(np.mean(scores))

def HYPTabs(AccBHP,X,y):
    CrAAtab1, CrAAtab2, CrAAtab3,CrAAtab4 = st.tabs(["K-Nearest Neighbors", "Decision Tree", "Gaussian Naive Bayes","Random Forest"])
    with CrAAtab1:
        classifier = make_pipeline(StandardScaler(),PCA(n_components=5),neighbors.KNeighborsClassifier(n_neighbors = 3))
        st.write(classifier.get_params())
        example4("Accuracy before Hyperparameter Tuning:")
        st.write(AccBHP[0])
        # The below should be used for grid search, but it takes so much time. Instead, to make app easy, I am reducing the parameters to already obtained best parameters.
        # k_range = list(range(1, 30))
        # ncs = list(range(1,11))
        Parameters = dict(kneighborsclassifier__n_neighbors=[11,16],pca__n_components =[5,6,8])
        grid = GridSearchCV(classifier, Parameters, cv = 5, scoring='accuracy')
        grid.fit(X, y)
        st.write(grid.best_params_)
        example4("Accuracy after Hyperparameter Tuning:")
        st.write(grid.best_score_)
    with CrAAtab2:
        classifier = make_pipeline(StandardScaler(),PCA(n_components=5),DecisionTreeClassifier(criterion = 'gini', random_state = 0))
        st.write(classifier.get_params())
        st.write(AccBHP[1])
        Parameters = dict(pca__n_components = [1,2,3],decisiontreeclassifier__criterion=['gini', 'entropy'],decisiontreeclassifier__max_depth=[2,3])
        grid = GridSearchCV(classifier, Parameters, cv = 3, scoring='accuracy')
        grid.fit(X, y)
        st.write(grid.best_params_)
        example4("Accuracy after Hyperparameter Tuning:")
        st.write(grid.best_score_)
    with CrAAtab3:
        classifier = make_pipeline(StandardScaler(),PCA(n_components=5),GaussianNB())
        st.write(classifier.get_params())
        st.write(AccBHP[2])
        example4("Gaussian Naive Bayes Classifier has limited parameters. Depending on the implementation, sometimes the number of classes is the only parameter, which in practice, we have no control on. So, Hyper-parameter tuning is not a valid method to improve Naive Bayes classifier accuracy. But here, We can implemet Hyperparameter tuning for PCA + Gaussian, to get best n components that gives best accuracy.")
        Parameters = dict(pca__n_components = list(range(1,11)))
        grid = GridSearchCV(classifier, Parameters, cv = 5, scoring='accuracy')
        grid.fit(X, y)
        st.write(grid.best_params_)
        example4("Accuracy after Hyperparameter Tuning:")
        st.write(grid.best_score_)
    with CrAAtab4:
        classifier = make_pipeline(StandardScaler(),PCA(n_components=5),RandomForestClassifier(random_state=0))
        st.write(classifier.get_params())
        st.write(AccBHP[3])
        Parameters = dict(pca__n_components = [2,10],randomforestclassifier__criterion=['gini'],randomforestclassifier__max_depth=[5,8],
        randomforestclassifier__max_features = ['auto'], randomforestclassifier__min_samples_leaf = [4],randomforestclassifier__min_samples_split = [10],
        randomforestclassifier__n_estimators = [200])
        grid = GridSearchCV(classifier, Parameters, cv = 3, scoring='accuracy')
        # grid = RandomizedSearchCV(classifier, Parameters, cv = 3, scoring='accuracy',n_jobs = -1,n_iter = 50,verbose=2)
        grid.fit(X, y)
        st.write(grid.best_params_)
        example4("Accuracy after Hyperparameter Tuning:")
        st.write(grid.best_score_)



if(Cr == "Hyperparameter Tuning"):
    example2("Hyperparameter Tuning:")
    example4("- Until now we have seen the classifiers with baseline model. But the parameters in those estimators define the nature of the model. These input parameters are called Hyperparameters.")
    example4("- The Hyperparameters will define the architecture of the model, and the best part about these is that you get a choice to select these for your model. Of course, you must select from a specific list of hyperparameters for a given model as it varies from model to model.")
    example4("- The selection procedure for hyperparameter is known as Hyperparameter Tuning. I used Random search and Grid search algorithms to find these Hyperparameters.")
    example4("Let's see the default parameters for our classifiers and try to find the best of those:")
    with st.expander("Beta Plasma Classification"):
        X, y = BetaPlasmaData(pdata)
        X_train, X_test, y_train, y_test = TrainTest(X,y)
        AccBHP = Accs(X_train, X_test, y_train, y_test)
        HYPTabs(AccBHP,X,y)
        
    with st.expander("Retinol Plasma Classification"):
        X, y = RetPlasmaData(pdata)
        X_train, X_test, y_train, y_test = TrainTest(X,y)
        AccBHP = Accs(X_train, X_test, y_train, y_test)
        HYPTabs(AccBHP,X,y)

    example4("After the hyperparameters are obtained, our goal is to use them in predicting the class for new data.")



if( Cr == "Prediction"):
    example2("Prediction:")
    example4("We have obtained the best model with best hyperparameters. Now we will use those to predict the class for new data.")
    example4("Give your data, and find whether your Beta plasma or Retinol levels are normal.")
    AGE = st.number_input("Age:",min_value = 1,step =1,value = 22)

    GENDER = st.selectbox("Gender:",("Male","Female"))
    if(GENDER == "Male"):
        SEX = 1
    else:
        SEX = 0

    STATofSMOKE = st.radio("Smoking Status:",("Never","Former Smoker","Current Smoker"))
    if(STATofSMOKE == "Never"):
        SMOKSTAT = 1
    elif(STATofSMOKE == "Former Smoker"):
        SMOKSTAT = 2
    else:
        SMOKSTAT = 3

    WEIGHT = st.number_input("Weight (Kgs):",step = 0.01,min_value = 0.10,value = 65.00)
    HEIGHT = st.number_input("Height (meters):",step = 0.01,min_value = 0.10, value = 1.78)
    QUETELET = WEIGHT/(HEIGHT * HEIGHT)

    VIT = st.radio("Vitamin-A Intake:",("Yes, fairly often", "Yes, not often", "No"))
    if(VIT == "Yes, fairly often"):
        VITUSE = 1
    elif(VIT == "No"):
        VITUSE = 3
    else:
        VITUSE = 2

    CALORIES = st.number_input("Number of Calories Consumed per day:",step = 1,value = 2000)

    FAT = st.number_input("Grams of Fat Consumed per day:",step = 0.1,value = 20.0)

    FIBER = st.number_input("Grams of Fiber Consumed per day",step = 0.1, value = 28.0)

    ALCOHOL = st.number_input("Number of Alchoholic Drinks Consumed per week:",step = 1,value = 0)

    CHOLESTEROL = st.number_input("Cholesterol consumed (mg per day):",step = 1,value = 300)

    BETADIET = st.number_input("Dietary Beta-Carotene Consumed (mcg per day):",step = 1,value = 2500)

    RETDIET = st.number_input("Dietary Retinol Consumed (mcg per day):",step = 1,value = 800)
    Ndata = [AGE,SEX,SMOKSTAT,QUETELET,VITUSE,CALORIES,FAT,FIBER,ALCOHOL,CHOLESTEROL,BETADIET,RETDIET]
    Newdata = pd.DataFrame(columns = ['AGE', 'SEX', 'SMOKSTAT', 'QUETELET','VITUSE', 'CALORIES', 'FAT', 'FIBER', 'ALCOHOL', 'CHOLESTEROL', 'BETADIET', 'RETDIET'])
    Newdata.loc[0] = Ndata

    example2("Estimation of Plasma Levels:")
    Prtab1, Prtab2 = st.tabs(["Regression","Classification"])

    with Prtab1:
        with st.expander("Beta-Carotene Plama Levels"):
            if(BETADIET == 0.00):
                st.markdown("Give valid Beta-carotene Consumption value (mcg per day) !")
            else:
                CrMs1 = st.multiselect("Select atleast a Feature:",['AGE', 'SEX', 'SMOKSTAT', 'QUETELET','VITUSE', 'CALORIES', 'FAT', 'FIBER', 'ALCOHOL', 'CHOLESTEROL', 'BETADIET'],['AGE'])
                x = np.array([pdata[CrMs1[0]]]).T
                ones = np.ones((x.shape[0],1))
                for i in range(1,len(CrMs1)):
                    x = np.append(x,np.array([pdata[CrMs1[i]]]).T,axis = 1)
                X = np.append(ones,x,axis =1)
                y = np.array([pdata['BETAPLASMA']]).T
                w = np.linalg.pinv(X.T @ X)@ X.T @ y
                preddata = Newdata[CrMs1]
                RP = w[0]
                for i in range(len(preddata.loc[0])):
                    RP = RP + w[i+1] * preddata.loc[0][i]
                example3("Your Data:")
                st.write(preddata)
                example3("Predicted value for Plasma Beta-Carotene (ng/ml):")
                st.write(RP[0])
        with st.expander("Retinol Plama Levels"):
            if(RETDIET == 0.00):
                st.markdown("Give valid Retinol Consumption value (mcg per day) !")
            else:
                CrMs1 = st.multiselect("Select atleast a Feature:",['AGE', 'SEX', 'SMOKSTAT', 'QUETELET','VITUSE', 'CALORIES', 'FAT', 'FIBER', 'ALCOHOL', 'CHOLESTEROL', 'RETDIET'],['AGE'])
                x = np.array([pdata[CrMs1[0]]]).T
                ones = np.ones((x.shape[0],1))
                for i in range(1,len(CrMs1)):
                    x = np.append(x,np.array([pdata[CrMs1[i]]]).T,axis = 1)
                X = np.append(ones,x,axis =1)
                y = np.array([pdata['RETPLASMA']]).T
                w = np.linalg.pinv(X.T @ X)@ X.T @ y
                preddata = Newdata[CrMs1]
                RP = w[0]
                for i in range(len(preddata.loc[0])):
                    RP = RP + w[i+1] * preddata.loc[0][i]
                example3("Your Data:")
                st.write(preddata)
                example3("Predicted value for Plasma Retinol (ng/ml):")
                st.write(RP[0])
    with Prtab2:
        with st.expander("Beta-Carotene Plama Levels"):
            if(BETADIET == 0.00):
                st.markdown("Give valid Beta-carotene Consumption value (mcg per day) !")
            else:
                example3("Choose a Classifier!!")
                example4("I suggest to choose K-Nearest Neighbors or Gaussian Naive Bayes, those yeilded best accuracy so far.")
                example4("The Classifiers below has Hyperparameters predefined as per Hyperparameter Tuning.")
                CrBsRb = st.selectbox("Choose a Classifier:",("K-Nearest Neighbors", "Decision Tree", "Gaussian Naive Bayes","Random Forest"),key = 1)
                if(CrBsRb == "K-Nearest Neighbors"):
                    Classifier_method = neighbors.KNeighborsClassifier(n_neighbors = 16)
                    clf = make_pipeline(StandardScaler(),PCA(n_components = 5),Classifier_method)
                elif(CrBsRb == "Decision Tree"):
                    Classifier_method = DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
                    clf = make_pipeline(StandardScaler(),PCA(n_components = 2),Classifier_method)
                elif(CrBsRb == "Gaussian Naive Bayes"):
                    Classifier_method = GaussianNB()
                    clf = make_pipeline(StandardScaler(),PCA(n_components = 5),Classifier_method)
                else:
                    Classifier_method = RandomForestClassifier(criterion='gini',max_depth=5,max_features = 'auto',min_samples_leaf = 4,min_samples_split = 10,n_estimators = 200)
                    clf = make_pipeline(StandardScaler(),PCA(n_components = 2),Classifier_method)
                
                NDD = [AGE,SEX,SMOKSTAT,QUETELET,VITUSE,CALORIES,FAT,FIBER,ALCOHOL,CHOLESTEROL,BETADIET]
                preddata = pd.DataFrame(columns = ['AGE', 'SEX', 'SMOKSTAT', 'QUETELET','VITUSE', 'CALORIES', 'FAT', 'FIBER', 'ALCOHOL', 'CHOLESTEROL', 'BETADIET'])
                preddata.loc[0] = NDD
                example3("Your Data:")
                st.write(preddata)
                X1, y1 = BetaPlasmaData(pdata)
                X_train, X_test, y_train, y_test = TrainTest(X1,y1)
                clf.fit(X_train,y_train)
                PREDD = clf.predict(preddata)
                example3("Predicted class for Plasma Beta-Carotene Levels:")
                st.write(PREDD[0])
                if(PREDD[0] == 0):
                    example5("Abnormal")
                else:
                    example5("Normal")
        
        with st.expander("Retinol Plama Levels"):
            if(RETDIET == 0.00):
                st.markdown("Give valid Beta-carotene Consumption value (mcg per day) !")
            else:
                example3("Choose a Classifier!!")
                example4("I suggest to choose Decision Tree classifier, it yeilded best accuracy so far.")
                example4("The Classifiers below has Hyperparameters predefined as per Hyperparameter Tuning.")
                CrBsRb = st.selectbox("Choose a Classifier:",("K-Nearest Neighbors", "Decision Tree", "Gaussian Naive Bayes","Random Forest"),key  = 2)
                if(CrBsRb == "K-Nearest Neighbors"):
                    Classifier_method = neighbors.KNeighborsClassifier(n_neighbors = 16)
                    clf = make_pipeline(StandardScaler(),PCA(n_components = 5),Classifier_method)
                elif(CrBsRb == "Decision Tree"):
                    Classifier_method = DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
                    clf = make_pipeline(StandardScaler(),PCA(n_components = 1),Classifier_method)
                elif(CrBsRb == "Gaussian Naive Bayes"):
                    Classifier_method = GaussianNB()
                    clf = make_pipeline(StandardScaler(),PCA(n_components = 1),Classifier_method)
                else:
                    Classifier_method = RandomForestClassifier(criterion='gini',max_depth=5,max_features = 'auto',min_samples_leaf = 4,min_samples_split = 10,n_estimators = 200)
                    clf = make_pipeline(StandardScaler(),PCA(n_components = 10),Classifier_method)
                
                NDD = [AGE,SEX,SMOKSTAT,QUETELET,VITUSE,CALORIES,FAT,FIBER,ALCOHOL,CHOLESTEROL,RETDIET]
                preddata = pd.DataFrame(columns = ['AGE', 'SEX', 'SMOKSTAT', 'QUETELET','VITUSE', 'CALORIES', 'FAT', 'FIBER', 'ALCOHOL', 'CHOLESTEROL', 'RETDIET'])
                preddata.loc[0] = NDD
                example3("Your Data:")
                st.write(preddata)
                X1, y1 = RetPlasmaData(pdata)
                X_train, X_test, y_train, y_test = TrainTest(X1,y1)
                clf.fit(X_train,y_train)
                PREDD = clf.predict(preddata)
                example3("Predicted class for Plasma Retinol Levels:")
                st.write(PREDD[0])
                if(PREDD[0] == 0):
                    example5("Abnormal")
                else:
                    example5("Normal")

    

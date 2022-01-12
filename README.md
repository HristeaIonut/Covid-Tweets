# Covid-Tweets

Studenti:
-
<ol>
  <li>Constantinescu George-Gabriel</li>
  <li>Hristea Ionut-Alexandru</li>
  <li>Lupu Cezar-Justinian</li>
  <li>Tache Radu-Ioan</li>
</ol>

Cerinta - Vaccinare Covid: extragere sentimente din mesaje tip tweet
-

- considerati un set de date de tip tweet;
- extrageți informații referitoare la sentimentele persoanelor fata de vaccinare;
- considerati un caz de studiu (ex: Romania);
- Referinte: 
<ol>
  <li>
  COVID-19 Vaccination Awareness and Aftermath: Public Sentiment Analysis on Twitter Data and Vaccinated Population Prediction in the USA https://www.mdpi.com/2076-3417/11/13/6128/htm
  </li>
  <li>Predicting vaccine hesitancy from area-level indicators: A machine learning approach  https://onlinelibrary.wiley.com/doi/epdf/10.1002/hec.4430 </li>
</ol>
  
Setul de date folosit
-

- link: https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset#files ;
- nume fisier: `corona_tweets_30.csv`;
- dimensiune: `392,847 tweets`;
- setul de date este format din doua coloane: id-urile tweet-urile si o valoare intre -1 si 1 ce reprezinta sentimentul; 

Observatii referitoare la date
-

- Din cauza limitarii date de Twitter de `900 requests / 15 minute`, dar si din cauza faptului ca un unele tweet-uri au fost sterse, am ajuns la `dimensiunea finala: 16838`;

Prelucrarea setului de date
-

- A fost folosit Twitter API pentru a transforma ID-urile in tweet-uri propriu-zise;
- Cuvintele din tweet-uri au fost tokenizate, fiecare cuvant fiind inlocuit cu un numar ce ii este asociat in mod unic, tinand cont de setul de date;
- Intervalul valorii sentimentului a fost schimbat din [-1;1] in {0,1} pentru a putea fi folosit in reteaua neuronala;
- Reteaua neuronala este formata din 3 straturi : un strat `Embedding(20000, 128)`, un strat `LSTM(128, dropout=0.2, recurrent_dropout=0.2)` si un strat `Dense(1, activation='sigmoid')`;
- Testarea a  fost facuta pe o baza de date constituita din 70% date de antrenament, respectiv 30% date de testare;

Rezultatele rularii
-

- Variabila `accuracy` reprezinta acuratetea pentru datele de antrenament;
- Variabila `val_accuracy` reprezinta acuratetea pentru datele de test;

<img 
          src="https://user-images.githubusercontent.com/69734986/149118900-765d8747-a71a-4d5c-94a0-e4573e4786a1.png"  width="1000px" height="auto">

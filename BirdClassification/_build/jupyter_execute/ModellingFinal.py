#!/usr/bin/env python
# coding: utf-8

# # Notebook 2 - Modellierung eines Classifiers durch Supervised Learning 
# von Annika Scheug und Oliver Schabe

# # 1. Einleitung

# Wie bereits in Notebook 1 erwähnt, soll in diesem Projekt ein Neuronales Netz modelliert werden, welches Bilder verschiedener Vogelarten klassifizieren kann (Multiclass Classification).  
# Hierfür verwenden wir Convolutional Neural Networks (CNN), da sich diese nach verbreiteter Meinung am besten zur Klassifizierung von Bildern in mehrere Klassen eignen.  
# Wie in Notebook 1 bereits beschrieben, sollen folgende Vogelarten klassifiziert werden, um Anspruch und Machbarkeit im Rahmen zu halten (mehr dazu in Notebook 1): 
# * Huhn
# * Kakadu
# * Adler
# * Flamingo
# * Strauß
# * Eule
# * Pinguin
# * Meise
# * Tucan
# 
# Für unser vollständig selbst entwickeltes Modell streben wir eine Accuracy von mind. 80% an.  
# Zum Vergleich sollen außerdem vortrainierte Netze herangezogen und für den hier beschriebenen Anwendungsfall optmiert werden. Hier wird mit einer höheren Trefferquote als bei dem vollständig selbst modellierten Netz gerechnet.  
#   
# **Task:** Die Aufgabe unseres Modells ist die Klassifikation, also Erkennung von neun definierten Vogelarten in Bildern.  
# 
# **Experience:** Das Modell soll aus mehreren Tausend Bildern lernen, auf denen jeweils ein Vogel der definierten Vogelarten abgebildet ist und welches entsprechend gelabelt ist.   
# 
# **Performance:** Das Modell soll an dem Kennwert Accuracy gemessen werden, da diese die intuitivste Metrik ist und uns sagt, wie viele der Testbilder richtig erkannt wurden. Hierbei ist jedoch wichtig, dass nicht nur die gesamte Accuracy des Modells, sondern auch Performance pro Klasse betrachtet wird. Dazu siehen wir Precision, Recall und F1-Score heran. Dadurch soll soll erkannt werden, falls einzelne Klassen, welche evtl. sogar noch unterrepräsentiert sind, außergewöhnlich schlecht erkannt werden.

# # 2. Setup

# Zunächst werden alle für das Projekt wichtigen Funktionalitäten importiert.

# In[1]:


import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import image_dataset_from_directory

import seaborn as sns

import matplotlib.pyplot as plt


# In[8]:


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Input,Dropout,Flatten,Conv2D,MaxPool2D,RandomFlip,RandomRotation,Rescaling
from tensorflow.keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import vgg16, xception
from keras.models import Model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from numpy import expand_dims
import operator


# In[9]:


from utilsJM import plot_confusion_matrix


# In[10]:


import warnings
warnings.filterwarnings("ignore") 


# In[11]:


# Für Visualisierung CNN
#pip install visualkeras
import visualkeras
from PIL import ImageFont


# # 3. Import of Datasets

# Da verschiedene Modelle mit verschiedenen Datasets (unterschiedliche Bilder, Bildgröße und Batch Size) getestet werden sollen, wird zunächst eine Funktion erstellt, welche die Bilder vom angegebenen Pfad in der angegebenen Auflösung und Batch Size in ein Trainings Datenset (train_ds) und ein Validation Datenset (val_ds) lädt.  
# Da die verwendete Keras util "image_dataset_from_directory" nur die Möglichkeit bietet, die eingelesenen Daten in zwei Datensets aufzuteilen, wird das Validation Datenset danach noch in Validation und Test Daten aufgeteilt.

# In[41]:


# Funktion zum einlesen der Bilder und erstellen der Datensets
def create_datasets(image_size, batch_size, directory):

    train_ds = image_dataset_from_directory(
        directory,
        label_mode="categorical",
        validation_split=0.3,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = image_dataset_from_directory(
        directory,
        label_mode="categorical",
        validation_split=0.3,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    return train_ds, val_ds


# Nach den ersten Tests fiel bereits auf, dass die Ergebnisse, welche mit der in Notebook 1 entwickleten Methode abgezogenenen Bildern erzeugt wurden, nicht zufriedenstellend sind.
# Um zu vergleichen, ob die schlechten Ergebnisse an den Bildern oder am entwickelten Modell liegen, werden im folgenden drei Datasets erzeugt. Mit diesen drei Datasets sollen die Modell trainiert und getestet werden:    
# 
# **Dataset 1:** Nur Bilder, welche in Notebook 1 erzeugt wurden  
# 
# **Dataset 2:** Bilder aus Notebook 1, mit Bilder Datensets aus dem Internet angereichert (https://images.cv/)  
# 
# **Dataset 3:** Nur Bilder Datensets aus dem Internet (https://images.cv/)  

# ### 3.1 Erzeugen von Dataset 1

# Als erstes wird das Dataset erzeugt, welches nur aus den in Notebook 1 generierten Bildern besteht. Dazu wird zunächst die Funktion mit den gewählten Parametern aufgerufen.  
# Für die Parameter image_size und batch_size wurden verschiedene Werte getestet (bspw. batch_size von 64 und 128 oder Auflösung von 32x32 und 128x128), die hier abgebildeten Werte stellen für die meisten Modelle die optimal gewählten Parameter dar. Falls für ein trainiertes Modell final andere Parameter gewählt wurden, so ist dies in der Beschreibung des Modells weiter unten vermerkt.  
# Das hier beschriebene Vorgehen ist identisch für Dataset 2 und Dataset 3, lediglich der Pfad zu den entsprechenden Bildern unterscheidet sich.

# In[42]:


# Aufruf der Funktion zum erstellen der Datasets Test und Validation
image_size1 = [64, 64]
batch_size1 = 32
train_ds1, val_ds1 = create_datasets(image_size1, batch_size1,"..\ProjectBirdClassification\Bilder1")


# Mit Hilfe von "element_spec" können wir die Struktur des gerade erzeugten datasets überprüfen:

# In[43]:


train_ds1.element_spec


# Als Nächstes wird aus der ersten Hälfte des Validation Datasets (val_ds) ein Test Dataset (test_ds) erstellt:

# In[44]:


test_ds1 = val_ds1.take(31)
val_ds1 = val_ds1.skip(31)

print('Batches for testing -->', test_ds1.cardinality())
print('Batches for validating -->', val_ds1.cardinality())


# Nun werden die Namen der Klassen ausgegeben.

# In[45]:


# Die Klassen sind in allen drei datasets gleich und müssen daher nur einmal erzeugt werden
class_names = train_ds1.class_names
class_names


# Nachfolgend iterieren wir über das Datenset, um aus den one hot encoded labeln einen Vektor mit einem integer Wert je Klasse zu gewinnen:

# In[46]:


y_train_class1 = list()
for images, labels in train_ds1:
    for i, label in enumerate(labels):
        y_train_class1.append(np.argmax(label, axis=None, out=None))


# Anschließend wird in einer weiteren Schleife gezählt, wie oft jede Klasse im Datenset enthalten ist:

# In[47]:


class_count1 = list()
for c in range(len(class_names)):
    class_count1.append(y_train_class1.count(c))
 
print(class_count1)


# In[48]:


# Plotten der gezälten Bilder
sns.barplot(x=list(range(9)), y=class_count1)
plt.title('Number of images per class')


# In den Trainingsdaten von Dataset 1 liegen je Klasse zwischen 470 und 526 Bilder vor. Die Anzahl der Bilder pro Klasse sind ungefähr gleich verteilt.

# ### 3.2 Erzeugen von Dataset 2

# Hier wird Dataset 2 erzeugt, welches aus den selbst erzeugten Bidler aus Notebook 1 besteht und mit Bilder Datasets aus dem Internet angereichert wurde (https://images.cv/).  
# Dataset 2 wird auf die gleiche Weise erzeugt wie Dataset 1, siehe Abschnitt 3.1.

# In[49]:


# Aufruf der Funktion zum erstellen der Datasets Test und Validation
image_size2 = [64, 64]
batch_size2 = 32
train_ds2, val_ds2 = create_datasets(image_size2, batch_size2, "..\ProjectBirdClassification\Bilder2")


# In[50]:


test_ds2 = val_ds2.take(68)
val_ds2 = val_ds2.skip(68)

print('Batches for testing -->', test_ds2.cardinality())
print('Batches for validating -->', val_ds2.cardinality())


# In[51]:


y_train_class2 = list()
for images, labels in train_ds2:
    for i, label in enumerate(labels):
        y_train_class2.append(np.argmax(label, axis=None, out=None))


# In[52]:


class_count2 = list()
for c in range(len(class_names)):
    class_count2.append(y_train_class2.count(c))
 
print(class_count2)


# In[53]:


# Plotten der gezälten Bilder
sns.barplot(x=list(range(9)), y=class_count2)
plt.title('Number of images per class')


# In den Trainingsdaten von Dataset 2 liegen mehr Bilder vor als in Dataset 1, da hier zusätzliche die Bilder von https://images.cv/ verwendet wurden. Auch die Anzahl der Bilder je Klasse weicht stärker voneinander ab.  
# Klasse 0 (Chicken/Huhn) zum Beispiel ist nur mit 737 Bildern vertreten, während für Klasse 7 (Tit/Meise) 1409 Bilder vorliegen.  
# Dabei besteht die Gefahr, dass Modelle auf Basis dieser Trainingsdaten biased gegenüber der stärker vertretenen Klassen sind oder schlechte Erkennung von schwach vertretenen Klassen in der gesamten Accuracy des Modells untergehen.  
# Da später jedoch auch Metriken je Klasse getrackt werden sollen, werden die Bilder alle als Traingsmaterial für das Modell beibehalten.

# ### 3.3 Erzeugen von Dataset 3

# Hier wird Dataset 3 erzeugt, welches nur aus den Bilder Datasets aus dem Internet besteht (https://images.cv/).  
# Dataset 3 wird auf die gleiche Weise erzeugt wie Dataset 1, siehe Abschnitt 3.1.

# In[54]:


# Aufruf der Funktion zum erstellen der Datasets Test und Validation
image_size3 = [64, 64]
batch_size3 = 32
train_ds3, val_ds3 = create_datasets(image_size3, batch_size3, "..\ProjectBirdClassification\Bilder3")


# In[55]:


test_ds3 = val_ds3.take(46)
val_ds3 = val_ds3.skip(46)

print('Batches for testing -->', test_ds3.cardinality())
print('Batches for validating -->', val_ds3.cardinality())


# In[56]:


y_train_class3 = list()
for images, labels in train_ds3:
    for i, label in enumerate(labels):
        y_train_class3.append(np.argmax(label, axis=None, out=None))


# In[57]:


class_count3 = list()
for c in range(len(class_names)):
    class_count3.append(y_train_class3.count(c))
 
print(class_count3)


# In[58]:


# Plotten der gezälten Bilder
sns.barplot(x=list(range(9)), y=class_count3)
plt.title('Number of images per class')


# Ähnlich wie bei Dataset 2 sind hier unterschiedlich viele Bilder je Label enthalten. Aus den selben Gründen wie bei Dataset 2 sollen die Bilder so behalten werden.

# # 4. Bilderexploration

# In diesem Kapitel geht es darum, ein Gefühl für die Daten zu bekommen, indem Bilder der verschiedenen Klassen visualisiert werden.  

# Hierfür wollen wir mit dem folgenden Code zunächst ein Bild jeder Klasse mit dem dazugehörigen Label anzeigen. Dafür wird in einer Schleife durch die ersten zwei Batches des Datasets iteriert und ein Bild geplotted, sofern nicht bereits in einer vorherigen Iteration bereits ein Bild dieser Klasse geplotted wurde. Dies realisieren wir mit der Liste *plotted*. Wird ein Bild einer Klasse geplotted, wird am enstprechenden Index der Klassennummer der Wert von *plotted* auf 'plotted' gesetzt. Bevor ein Bild geplotted wird, wird in einer if-Bedingung geprüft, ob diese Klasse bereits geplotted wurde. Falls nicht, wird das Bild geplotted und anschließend *plotted* upgedated, falls ja wird mit dem nächsten Element weitergemacht.

# In[24]:


plt.figure(figsize=(10, 10))
pos=0
plotted = list(range(10))
for images, labels in train_ds1.take(2):
    for i in range(batch_size1):
        if(plotted[np.argmax(labels[i], axis=None, out=None)] != 'plotted'):
            ax = plt.subplot(3, 3, pos + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i], axis=None, out=None)])
            plt.axis("off")
            plotted[np.argmax(labels[i], axis=None, out=None)] = 'plotted'
            pos=pos+1        
        else:
            continue


# Die oben beschriebene Visualisierung eines Bilds jeder Klasse wird für das dataset 2 wiederholt.

# In[25]:


plt.figure(figsize=(10, 10))
pos=0
plotted = list(range(10))
for images, labels in train_ds2.take(2):
    for i in range(batch_size3):
        if(plotted[np.argmax(labels[i], axis=None, out=None)] != 'plotted'):
            ax = plt.subplot(3, 3, pos + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i], axis=None, out=None)])
            plt.axis("off")
            plotted[np.argmax(labels[i], axis=None, out=None)] = 'plotted'
            pos=pos+1        
        else:
            continue


# In beiden Plots ist zu erkennen, dass es deutliche Unterschiede der Klassenmerkmale gibt. Die Interclass Differenz, also die Unterschiede in Merkmalen verschiedener Klassen, ist dementsprechend ausreichend vorhanden. Beispielsweise sind Tucans deutlich an ihrem großen, bunten Schnabel zu erkennen, während Cockatoos einen auffälligen Federschmuck am Kopf besitzen.  
# Ein Unterschied zwischen den beiden Datasets ist zunächst nicht zu erkennen.

# Beim genaueren Untersuchen fällt auch auf, dass die in Notebook 1 erzeugten Bilder unterschiedliche Formate besitzen. Im Gegensatz zu den quadratischen Bildern aus den Datensets von https://images.cv/ tauchen hier Bilder in Hochformat und Querformat auf:

# **selbst erzeugtes Bild in Hochformat:**

# In[29]:


# Laden eines selbst erzeugten Bildes mit Hochformat
tf.keras.utils.load_img(r"..\ProjectBirdClassification\Bilder1\chicken\chicken (48).png")


# **selbst erzeugtes Bild in Querformat:**

# In[28]:


# Laden eines selbst erzeugten Bildes mit Querformat
tf.keras.utils.load_img(r"..\ProjectBirdClassification\Bilder1\chicken\chicken (51).png")


# **Bild von https://images.cv/ in quadratischen Format:**

# In[27]:


# Laden eines Bilds von https://images.cv/ in qudratischem Format
tf.keras.utils.load_img(r"..\ProjectBirdClassification\Bilder3\chicken\0K3D5SDTO0RJ.jpg")


# Beim einlesen solcher Bilder in ein quadratisches Format können somit Verzerrungen entstehen, welche für das Model ein Problem darstellen und ggfs. Einfluss auf die Performance haben könnten. Dies gilt es später zu beachten bzw. zu untersuchen.

# Im folgenden Plot sollen 9 Bilder einer Klasse visualisiert werden. Dafür wird in der Variablen *plot_class* zunächst die ausgewählte Klasse und in *class_index* deren Index definiert.  
# Anschließend iterieren wir wieder durch das Dataset und plotten ein Bild nur, wenn das Label der zuvor definierten Klasse entspricht.

# In[200]:


plot_class = "tucan"
class_index = class_names.index(plot_class)

plt.figure(figsize=(15, 15))
pos=0
for images, labels in train_ds3.take(10):
    for i in range(9):
      if(pos<9):
        if(np.argmax(labels[i], axis=None, out=None) == class_index):
            ax = plt.subplot(3, 3, pos + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(plot_class)
            plt.axis("off")
            pos=pos+1        
        else:
            continue
      else:
        break


# Die Bilder zeigen den Tucan immer aus anderen Winkeln oder Positionen. Allerdings sind dessen Hauptmerkmale (großer, bunter Schnabel, gelbe Brust, sonst hauptsächliches schwarzes Gefieder) in allen Bildern ausreichend vorhanden. Die Intraclass Varianz, also die Unterschiede innerhalb einer Klasse, sind daher eher klein. Zum Vergleich soll als nächste eine Klasse geplottet werden, welche vermutlich eine höhere Intraclass Varianz aufweist.

# In[376]:


plot_class = "owl"
class_index = class_names.index(plot_class)

plt.figure(figsize=(15, 15))
pos=0
for images, labels in train_ds3.take(20):
    for i in range(9):
      if(pos<9):
        if(np.argmax(labels[i], axis=None, out=None) == class_index):
            ax = plt.subplot(3, 3, pos + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(plot_class)
            plt.axis("off")
            pos=pos+1        
        else:
            continue
      else:
        break


# Bei der Klasse Eule können wir erkennen, dass innerhalb der Klasse größere optische Unterschiede vorhanden sind als bei der Klasse Tucan (beispielsweise Schneeeule, Schleiereule, Uhu etc.):

# **Schleiereule:**

# In[30]:


tf.keras.utils.load_img(r"..\ProjectBirdClassification\Bilder3\owl\0JQ0CI04VUGY.jpg")


# **Schneeeule:**

# In[31]:


tf.keras.utils.load_img(r"..\ProjectBirdClassification\Bilder3\owl\0NL38ZA7X622.jpg")


# **Uhu:**

# In[32]:


tf.keras.utils.load_img(r"..\ProjectBirdClassification\Bilder3\owl\0SKMABBW4H2K.jpg")


# Dies könnte bedeuten, dass die richtige Klassifikation dieser Klasse für das Modell schwieriger ist und somit schlechtere Ergebnisse liefert.  
# Die soll später bei der Evaluation der Modelle überprüft werden.

# # 5. Image Augmentation

# Um mit einer Bildklassifikation mithilfe von Neuronalen Netzen gute Ergebnisse zu erzielen, ist eine große Menge an Trainingsdaten notwendig. Nur so kann das Modell ausreichend trainiert werden, um die Merkmale unterschiedlicher Klassen gut zu unterscheiden.  
# Liegen nicht genügend Trainingsdaten vor, kann durch Image Augmentation die Datenmenge künstlich vergrößert werden.  
# Dafür werden vorhandene Bilder zufällig gedreht, gespiegelt und vergrößert bzw. verkleinert und diese neu generierten Bilder dem Modell als neue Bilder zur Verfügung zu stellen.

# Keras stellt hierfür verschiedene Funktionen zur Verfügung. Mit *RandomFlip* wird ein Bild zufällig (auswählbar ob horizontal, vertikal oder beides) gespiegelt, während *RandomRotation* ein Bild um einen gewissen Faktor zufällig dreht. Zusätzlich existiert beispielsweise auch der Layer *RandomZoom*, unter dessen Verwendung ein Bild zufällig vergrößert wird.  
# 
# Nachfolgend demonstrieren wir die Methoden *RandomFlip* und *RandomRotation* mit unserem Datenset.  
# Hierfür können die beiden Preprocessing Layer in einem Sequential Objekt kombiniert werden, sodass diese sich wie eine Funktion mit dem Namen *data_augmentation* aufrufen lassen. 

# In[31]:


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ]
)


# Das Ergebnis dieser Augmentation visualisieren wir wieder in einem Plot, indem wir durch das Datenset iterieren und zunächst das Originalbild und anschließend das augmentierte Bild anzeigen. Die Augmentation erfolgt, indem die Originalbilder *images* an die eben erzeugte Funktionalität *data_augmentation* übergeben werden. Dadurch wird jedes Originalbild einmal indivduell zufällig gespiegelt und rotiert.

# In[32]:


plt.figure(figsize=(15, 15))
for images, labels in train_ds1.take(1):
    augmented_images = data_augmentation(images)
    for i in range(0, 15, 2):
        axarr = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title("original: " + class_names[np.argmax(labels[i], axis=None, out=None)])
        axarr = plt.subplot(4, 4, i + 2)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title("augmented: " + class_names[np.argmax(labels[i], axis=None, out=None)])
        plt.axis("off")



# Es ist zu erkennen, dass jedes Orignalbild mit einem anderen Faktor gedreht wurde und auch jedes Originalbild anders gespiegelt wurde.

# Die Image Augmentation kann entweder zu Beginn auf das Datenset angewendet werden und deren Ergebnisse persistiert und dem Datenset hinzugefügt werden. Somit vergrößert sich natürlich die Datenmenge. In einer Trainingsepoche des Neuronalen Netzes stehen mehr Bilder zur Verfügung, als wenn nur Originalbilder verwendet werden.    
# Alternativ besteht die Möglichkeit, die eben demonstrierten Preprocessing Layer der Architektur eines Neuronalen Netzes hinzuzufügen. Somit wird in der Trainingsphase jedes Bild zunächst augmentiert, bevor es den Rest des Modells durchläuft. Dadurch vergrößert sich zwar nicht die Datenmenge innerhalb einer Modellepoche, da jedes Bild genau einmal das neuronale Netz durchläuft. In der nächsten Epoche wird das Bild aber wieder augmentiert, um einen zufälligen Faktor. Somit erscheint es für das Modell als neues Bild.  
# Vergrößern wir also die Anzahl der Epochen, im Vergleich zu Alternative 1, so erzielen wir auf lange Sicht den gleichen Effekt und vergrößern unser Datenset. Für diese Alternative haben wir uns in unserem Projekt entschieden, weshalb die Ergebnisse der demonstrierten Image Augmentation nicht gespeichert werden. Stattdessen fügen wir die beiden Preprocessing Layer *RandomFlip* und *RandomRotation* unseren Modellen hinzu.

# # 6. Building & Training a CNN Model from Scratch

# Da sich wie bereits erwähnt CNN am besten zur Multiclass Classification von Bildern eignen, sollen in diesem Kapitel verschiedene CNN Architekturen selbst erstellt und getestet werden. Durch das Designen einer geeigneten Architektur sowie das Auswählen optimaler Parameter soll eine möglichst hohe Accuracy erzielt werden.  
# Wir beginnen mit einem simplen Modell und steigern die Komplexität der Architektur & Parameter von Modell zu Modell.

# ## 6.1 Model 1 - simples CNN

# ### a) Model 1 - Introduction

# In diesem ersten Versuch soll ein simples CNN mit wenigen Layern und wenigen Filtern erstellt werden. Dies soll als Test dienen, welche Ergebnisse mit einem einfachen Netz und wenig Ressourcen erzielt werden können.  
# Hier soll der Fokus noch nicht auf der Optimierung der Architektur und der Hyperparameter liegen. Dies erfolgt später in Netzen mit komplexerer Architektur, welche auch mehr Aussicht auf Erfolg bieten.  
# Außerdem bietet sich dieses Netz an, um die drei unterschiedlichen Datasets durchzutesten und zu vergleichen, da das CNN auf Grund der einfachen Architektur vergleichsweise schnell trainiert werden kann.

# ### b) Model 1 - Construction

# Um möglichst einfach mehrere gleiche Modelle zum Testen der verschiedenen Datasets erstellen zu können, wird die Konstruktion des Modells in eine Funktion ausgelagert, welche immer wieder aufgerufen werden kann um ein gleiches CNN zu erstellen.

# In[99]:


def construct_model1():
    cnn1 = Sequential()
    cnn1.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64,3), padding='same',activation='relu'))
    cnn1.add(Dropout(0.2))
    cnn1.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',padding='same'))
    cnn1.add(MaxPool2D(pool_size=(2, 2)))
    cnn1.add(Flatten())
    cnn1.add(Dense(512, activation='relu'))
    cnn1.add(Dropout(0.5))
    cnn1.add(Dense(len(class_names), activation='softmax'))
    return cnn1


# **Architektur und Parameter:**  
# An dieser Stelle wurde nicht viel Wert auf optimale Architektur oder Parameter gelegt. Layer, Filter und weitere Parameter wurden lediglich so gewählt, dass das Training des Modells einigermaßen schnell auf lokaler Hardware durchführbar ist um möglichst schnell ein erstes Gefühl für die Verwertbarkeit der verschiedenen Datasets zu bekommen sowie für den Accuracy Wert eines nicht optimierten, simplen CNN.  
# Die Bilder zum Trainieren wurden für diese Modelle mit einer Auflösung von 64x64 Pixel eingelesen, da nach Tests keine Auswirkung der Auflösung im Gegensatz zu 128x128 festgestellt werden konnte und mit 64x64 Pixel das Modell schneller trainiert werden kann.

# Als nächstes werden nun drei Modelle erstellt, welche mit den drei unterschiedlichen datasets trainiert werden sollen, um Unterschiede zwischen den datasets zu untersuchen.  
# Das erste Modell soll mit Dataset 1 trainiert werden, das zweite Modell mit Dataset 2 und das dritte Modell mit Dataset 3.

# In[100]:


# Erstellen des ersten simplen CNN Modells
model1_1=construct_model1()
model1_1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[111]:


# Erstellen des zweiten simplen CNN Modells
model1_2=construct_model1()
model1_2.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[112]:


# Erstellen des dritten simplen CNN Modells
model1_3=construct_model1()
model1_3.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# Bevor die Modelle trainiert werden, können wir die Layer unserer simplen CNN Architektur mit Hilfe einer visualkeras util visualisieren:

# In[103]:


# Visuelle Darstellung der Layer von Modell 1, gleich für alle drei erstellten Modelle
visualkeras.layered_view(model1_1, legend=True)


# In[109]:


model1_1.summary()


# ### c) Model 1 - Training

# Nun werden die zuvor erstellten Modelle mit den jeweiligen Datasets trainiert.

# In[104]:


# Trainieren des einfachen CNN mit dataset 1 (in Notebook 1 erzeugte Bilder)
print("Trainieren des einfachen CNN mit dataset 1 (in Notebook 1 erzeugte Bilder)")
history1_1 = model1_1.fit(train_ds1, batch_size=32, epochs=12, verbose=True, validation_data=val_ds1)


# In[113]:


# Trainieren des einfachen CNN mit dataset 2 (in Notebook 1 erzeugte Bilder + Bilder von https://images.cv/)
print("Trainieren des einfachen CNN mit dataset 2 (in Notebook 1 erzeugte Bilder + Bilder von https://images.cv/)")
history1_2 = model1_2.fit(train_ds2, batch_size=32, epochs=12, verbose=True, validation_data=val_ds2)


# In[114]:


# Trainieren des einfachen CNN mit dataset 3 (nur Bilder von https://images.cv/)
print("Trainieren des einfachen CNN mit dataset 3 (nur Bilder von https://images.cv/)")
history1_3 = model1_3.fit(train_ds3, batch_size=32, epochs=12, verbose=True, validation_data=val_ds3)


# Am Trainingsverlauf können wir bereits erkennen, dass sich Dataset 2 und Dataset 3 am besten als Input für das Modell eignen.

# ### d) Model 1 Evaluation

# Für die Evaluation der Modelle ist zunächst deren Trainings- und Validationaccuracy sowie der Loss interessant.  
# Hierfür wird die Funktion *plot_accuracy* definiert, welche den Verlauf dieser beiden Werte für Trainings- und Validierungsdaten über alle Epochen visualisiert. Als Input benötigt diese Funktion die Modelhistory, welche in der Trainingsphase des jeweiligen Modells erstellt wurde. Aus dieser Historie können die einzelnen Epochen Werte von Accuracy und Loss sowie deren Maximum bzw. Minimum bestimmt und anschließend geplotted werden.

# In[115]:


# Erstellen einer Funktion zum plotten der Accuracy
def plot_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    max_val_acc=np.max(val_acc)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    min_val_loss=np.min(val_loss)

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(16,8))
    ax = plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    print("Maximum accuracy: ",max_val_acc)
    print("Minimum Loss: ",min_val_loss)



    ax = plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# In[116]:


print("Accuracy und Loss des simplen CNN mit dataset 1:")
plot_accuracy(history1_1)

print("Accuracy und Loss des simplen CNN mit dataset 2:")
plot_accuracy(history1_2)

print("Accuracy und Loss des simplen CNN mit dataset 3:")
plot_accuracy(history1_3)


# Wie bereits zuvor schon vermutet, lässt sich an Hand der hier visualisierten Metriken noch einmal bestätigen, dass die Datasets 2 und 3 beim gleichen Modell eine bessere Performance liefern als Dataset 1. Der Trainingsverlauf an sich scheint einigermaßen stabil, die Validation Accuracy steigt von Epoche zu Epoche meist leicht an. Vermutlich hätte durch weitere Epochen ein noch besseres Ergebnis erzielt werden können. Da dieses Modell allerdings nur ein erster Test sein sollte, belassen wir es bei dem oben dargestellten Training.

# Mit Accuracy und Loss alleine kann, wie zu Beginn des Notebooks beschrieben, keine ausreichende Aussage über Performance eines Modells getroffen werden. Daher wollen wir nun auch Precision, Recall und F1-Score der Modelle betrachten.  
# Hierfür definieren wir zunächst die Funktion *predict_testdata*, die für das übergebene Modell und Testdaten y_true und y_pred bestimmt. Dafür iteriert die Funktion über die Testdaten und bestimmt für jeden Batch die jeweils richtige und die durch das Modell vorhergesagte Klasse. Zuletzt werden die einzelnen Batchelemente von y_true und y_pred in jeweils ein Tensorobjekt konvertiert und dieses als Output der Funktion zurückgegeben.

# In[76]:


#Funktion berechnet y_pred und y_true für ein Modell, daher Modell und richtiges test_ds mitgeben, als output kommt y_true und y_pred
def predict_testdata(model, test_ds):
    
   y_pred = []  
   y_true = []  

   for image_batch, label_batch in test_ds:   
      y_true.append(np.argmax(label_batch, axis=-1))
      preds = model.predict(image_batch)
      y_pred.append(np.argmax(preds, axis = - 1))

   # Konvertierung zu Tensor Objekten
   correct_labels = tf.concat([item for item in y_true], axis = 0)
   predicted_labels = tf.concat([item for item in y_pred], axis = 0)

   return correct_labels, predicted_labels


# Diese Funktion wird nun beispielhaft für das beste der oben trainierten Modelle (model1_3) und test_ds3 aufgerufen und die beiden Ergebnisse in den Variablen model1_3_corL (für correctLabel) und model1_3_predL (für predictedLabel) gespeichert.

# In[118]:


#Aufruf der Funktion
model1_3_corL, model1_3_predL = predict_testdata(model1_3, test_ds3)


# Mithilfe der eben erzeugten Variablen können die Metriken Precision, Recall und F1-Score nun für das Modell angezeigt werden.

# In[119]:


print(classification_report(model1_3_corL, model1_3_predL,target_names = class_names))


# Anschließend wird die Confusion Matrix des Modells mithilfe der importieren Funktion *plot_confusion_matrix* erstellt.

# In[120]:


plot_confusion_matrix(confusion_matrix(model1_3_corL, model1_3_predL))


# An Classification Report und Confusion Matrix lässt sich leicht erkennen, dass das Modell 1 nicht in der Lage ist korrekte Klassifizierungen durchzuführen. Vor allem die Klasse Owl wird nur sehr schlecht erkannt, und viele Bilder werden falsch als Flamingo klassifiziert.

# ### e) Model 1 - Fazit

# In Modell 1 können wir bereits erkennen, dass sich Dataset 1 nicht besonders gut eignet. Daher sollen in den weiteren Modellen nurnoch Dataset 2 und Dataset 3 verwendet werden. Dataset 1 beinhaltet vermutlich zu wenige Bilder in guter Qualität. Außerdem könnte das zuvor erwähnte Problem mit Hoch- und Querformat der originalen Bilder eine Rolle spielen. 
#   
# Die Performance von Modell 1 ist noch nicht optimal, der Fokus auf Optimierung der Architektur und Parameter erfolgt allerdings auch erst in Modell 2.

# ## 6.2 Model 2 - komplexeres CNN

# ### a) Model 2 - Introduction

# Mit den bei Modell 1 gewonnen Erkentnissen soll nun ein zweites Modell designed werden, welches mehr Layer und Filter besitzt als das vorherige und damit auch bessere Ergebnisse erzielt. Der Anspruch ist es, hierbei eine Konfiguration zu finden, welche sich in angemessener Zeit auf der lokalen Hardware trainieren lässt.

# ### b) Model 2 - Construction

# Wie bereits in Modell 1 wird das Design des Modells in eine Funktion ausgelagert, um schnell mehrere Modelle erstellen und testen zu können. Zusätzlich wird hier noch die Learning Rate in die Funktion als Parameter mitgegeben, um diese beim Aufrufen der Funktion zu Testzwecken einfach ändern zu können.

# In[337]:


def construct_cnn2(learningRate):
    cnn2 = Sequential()
    cnn2.add(RandomFlip("horizontal_and_vertical", input_shape=(64, 64,3)))
    cnn2.add(RandomRotation(0.2))
    cnn2.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    cnn2.add(Dropout(0.1))
    cnn2.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    cnn2.add(Dropout(0.1))
    cnn2.add(MaxPool2D((2, 2)))
    cnn2.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    cnn2.add(Dropout(0.1))
    cnn2.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    cnn2.add(Dropout(0.1))
    cnn2.add(MaxPool2D((2, 2)))
    cnn2.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    cnn2.add(Dropout(0.1))
    cnn2.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    cnn2.add(Dropout(0.1))
    cnn2.add(MaxPool2D((2, 2)))
    cnn2.add(Flatten())
    cnn2.add(Dense(256, activation='relu'))
    cnn2.add(Dense(256, activation='relu'))
    cnn2.add(Dense(len(class_names), activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    cnn2.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return cnn2


# In diesem Modell wurden viele Parameter und Architekturen getestet:

# **Architektur und Parameter:**  
# **Batch Size:** 32 hat sich durch Tests als effizient erwiesen  
# **Auflösung der Bilder**: 32x32 Pixel erzielte weniger gute Ergebnisse, zwischen 64x64 und 128x128 konnten in mehreren Probedurchläufen keine ausschlaggebenden Unterschiede festegestellt werden. Aus Performance Gründen wurde daher 64x64 Pixel gewählt.  
# **Anzahl Layer:**  Bei der Anzahl an Layer wurde durch Tests versucht, ein Optimum zwischen Geschwindigkeit und Ergebnis zu finden. Zur Orientierung für den Aufbau der Layer wurden best practice Architekturen aus dem Internet, bspw. GitHub herangezogen.  
# **Anzahl Filter:** Hier wurde als best practice eine von Layer zu Layer aufsteigende Anzahl an Filtern getestet zwischen 32 und 256. Die beste Ergebnisse wurden wie oben dargestellt erzielt.  
# **Filtergröße:** 3x3 als Standard bei Bildern kleiner 128x128  
# **Dropout:** ein Dropout mit geringer Größe nach jedem convolutional Layer hat sich als wirkungsvoll erwiesen, empfohlen aus Forschung/Literatur (http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf)<br> 
# **Augmentation:**  Mit den Keras Layern Random Flip und Random Rotation konnten etwas bessere Ergebnisse erzielt werden.  
# **Activation:** Als Aktivierungsfunktion für die Hidden Layer wurden relu und tanh als CNN Standards getestet, mit relu konnten die besseren Ergebnisse erzielt werden. Für den Ausgabe Layer wurde softmax als Standard gewählt.  
# **Optimizer:**  Adam und SGD wurden getestet, wobei keine merkbaren Unterschiede aufgefallen sind, daher wurde Adam gewählt.  
# **Learning Rate:** Learning Rate von 0,001 bewährt, liefert ein "stabileres" Training mit weniger Schwankungen in Validation Accuracy als bspw. 0,01  
# **Loss:** Categorical Crossentropy als Standard für multiclass classification CNN.
# 

# Da sich Dataset 1 als weniger geeignet erwiesen hat, werden nun nurnoch zwei gleiche Modelle erstellt, welche mit Dataset 2 und 3 trainiert werden sollen:

# In[350]:


# Erstellen des ersten komplexeren CNN Modells, welches mit Dataset 2 trainiert werden soll:
Model2_1=construct_cnn2(0.001)
Model2_1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[338]:


# Erstellen des zweiten komplexeren CNN Modells, welches mit Dataset 3 trainiert werden soll:
Model2_2=construct_cnn2(0.001)
Model2_2.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# Auch lässt sich gut erkennen, dass Modell 2 aus mehr Schichten besteht als Modell 1:

# In[327]:


# Visuelle Darstellung der Layer von Modell 2, gleich für alle zwei erstellten Modelle
visualkeras.layered_view(Model2_1, legend=True)


# In[383]:


Model2_1.summary()


# ### c) Model 2 - Training

# In[351]:


history2_1 = Model2_1.fit(train_ds2, batch_size=32, epochs=40, verbose=True, validation_data=val_ds2)


# In[339]:


history2_2 = Model2_2.fit(train_ds3, batch_size=32, epochs=40, verbose=True, validation_data=val_ds3)


# Das Modell mit den besten identifizierten Parametern (so wie oben konstruiert) wird nun abgespeichert:

# In[56]:


# Speichern des Models
#Model2_2.save('CNN2best.h5')


# Das gespeicherte Modell kann nun jederzeit wieder geladen werden. Die Evaluation erfolgt auf Basis des abgespeicherten Modells, welches mit Dataset 3 trainiert wurde.

# In[33]:


Model2Best=load_model('CNN2best.h5')


# Mit Hilfe des folgenden Codes können wir uns die Feature Maps eines bestimmten Layers visualisieren lassen.

# In[80]:


# definiere modell als output nach ersten hidden layer
model = Model(inputs=Model2Best.inputs, outputs=Model2Best.layers[2].output)
Model2Best.summary()
# Bild mit richtiger Image Size laden
img = load_img(r"..\ProjectBirdClassification\Bilder2\chicken\0K3D5SDTO0RJ.jpg", target_size=(64, 64))
# Bild zu Array konvertieren
img = img_to_array(img)
# Dimension expandieren
img = expand_dims(img, axis=0)
# erzeugen der Feature Maps für ausgewählten Layer
feature_maps = model.predict(img)
# alle Feature Maps plotten
square = 8
ix = 1
plt.figure(figsize=(20,20))
for _ in range(8):
	for _ in range(4):
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
pyplot.show()


# An den geplotetten Feature Maps können wir bereits erkennen, welche Merkmale das Modell in dem angegebenen Layer (in diesem Fall Layer 2, erster convolutional Layer) extrahiert.  
# Zum Vergleich printen wir hier das normale Bild:

# In[39]:


tf.keras.utils.load_img(r"..\ProjectBirdClassification\Bilder2\chicken\0K3D5SDTO0RJ.jpg")


# ### d) Model 2 Evaluation

# In[356]:


print("Accuracy und Loss des komplexeren CNN mit dataset 2:")
plot_accuracy(history2_1)

print("Accuracy und Loss des komplexeren CNN mit dataset 3:")
plot_accuracy(history2_2)


# Hier fällt auf, dass beim Training teilweise stärkere Schwankungen in der Validation Accuracy auftreten. Dies liegt vermutlich daran, dass das Modell noch relativ häufig am raten ist.

# In[81]:


model2_corL, model2_predL = predict_testdata(Model2Best, test_ds3)


# In[82]:


print(classification_report(model2_corL, model2_predL,target_names = class_names))


# In[64]:


plot_confusion_matrix(confusion_matrix(model2_corL, model2_predL))


# die zwei Klassen mit dem schlechsten F1-Score sind hier cockatoo und owl:
# Bei der Klasse cockatoo fällt auf, dass die Precision sehr hoch ist (fast alle der als cockatoo klassifizierten Bilder sind tatsächlich cockatoos), allerdings der Recall sehr schlecht ist (nur wenige der als cockatoo wahr klassifierten Testbilder wurden als solche erkannt).  
#   
# Bei der Klasse owl dagegen sind sowohl Precision als auch Recall schlecht. Wie bereits in Kapitel 4 vermutet, könnte dies an der relativ großen Intraclass Varianz liegen, welche zuvor erörtert wurde.  

# Zum Abschluss wählen wir noch einmal komplett neue Bilder aus, um diese vom Modell klassifizieren zu lassen. Dazu wird die Funktion *predict_image* erstellt und auf die Bilder angewendet.

# In[83]:


def predict_image(directory, image_size, model):

    # Laden des Bilds in der angegebenen Größe
    img = load_img(directory, target_size=image_size)
    img = img_to_array(img)
    plt.imshow(img.astype("uint8"))
    img = expand_dims(img, axis=0)

    # Klassifikation durchführen
    predictions = model.predict(img)

    dic = dict(zip(class_names, predictions[0]))

    #Sortieren der Vorhersagen nach Wahrscheinlichkeit
    sorted_d = dict( sorted(dic.items(), key=operator.itemgetter(1),reverse=True))
    print(sorted_d)


# In[88]:


predict_image("..\ProjectBirdClassification\Testbilder\madagascar-penguin.jpg", [64, 64], Model2Best)


# **Originalbild:**

# In[91]:


tf.keras.utils.load_img(r"..\ProjectBirdClassification\Testbilder\madagascar-penguin.jpg")


# In[90]:


predict_image("..\ProjectBirdClassification\Testbilder\TucanHaribo.jpg", [64, 64], Model2Best)


# **Originalbild:**

# In[93]:


tf.keras.utils.load_img("..\ProjectBirdClassification\Testbilder\TucanHaribo.jpg")


# In[116]:


predict_image("..\ProjectBirdClassification\Testbilder\Hedwig.jpg", [64, 64], Model2Best)


# **Originalbild:**

# In[117]:


tf.keras.utils.load_img("..\ProjectBirdClassification\Testbilder\Hedwig.jpg")


# Der Pinguin und der Tucan konnten vom Modell korrekt erkannt werden, die Eule wurde allerdings evtl. auf Grund des bunten Schals als Tucan klassifiziert.

# ### e) Model 2 - Fazit

# Wie erwartet performt Modell 2 auf Grund der komplexeren Architektur und den vielen erprobten Parameteroptionen und Kombinationen besser als Modell 1. Das Ziel von 80% Accuracy konnte allerdings nicht erreicht werden. Dies liegt vermutlich an einer Kombinationen von nicht perfekten Trainingsdaten (Zu wenige Bilder, teilweise hohe Varianz in Klassen) und beschränkter Ressourcen zum Trainieren auf dem lokalen Rechner.  
#   
# Um den letzten Punkt zu prüfen, soll noch ein weiteres Modell mit noch komplexerer Architektur erstellt werden, um zu testen ob dies die Performance weiter verbessern kann.

# ## 6.3 Model 3

# ### c) Model 3 - Introduction

# Als letztes selbst erstelltes Modell soll nun mit dem zuvor erworbenen Wissen ein CNN mit noch mehr Layern und Filtern erstellt werden um zu überprüfen, ob dies die Performance weiterhin optimieren kann. Dazu wird die in Modell 2 bewährte Architektur um einen zusätzlichen convolutional Layer ergänzt.

# ### a) Model 3 - Construction

# In[95]:


def construct_cnn3(learningRate):
    cnn3 = Sequential()
    cnn3.add(RandomFlip("horizontal_and_vertical", input_shape=(64, 64,3)))
    cnn3.add(RandomRotation(0.2))
    cnn3.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    cnn3.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPool2D((2, 2)))
    cnn3.add(Dropout(0.2))
    cnn3.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    cnn3.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPool2D((2, 2)))
    cnn3.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    cnn3.add(MaxPool2D((2, 2)))
    cnn3.add(Conv2D(filters=1024, kernel_size=(3, 3), activation='relu'))
    cnn3.add(Flatten())
    cnn3.add(Dense(512, activation='relu'))
    cnn3.add(Dropout(0.2))
    cnn3.add(Dense(len(class_names), activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    cnn3.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return cnn3


# **Architektur und Parameter:**  
# Für dieses Modell werden die zuvor als optimal getesteten Parameter gewählt und lediglich mit dem hinzufügen von zusätzlichen Layern experimentiert.

# In[96]:


Model3=construct_cnn3(0.001)


# In[97]:


Model3.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[98]:


# Visuelle Darstellung der Layer von Modell 3
visualkeras.layered_view(Model3, legend=True)


# In[282]:


Model3.summary()


# ### b) Model 3 - Training

# In[283]:


history3 = Model3.fit(train_ds3, batch_size=32, epochs=20, verbose=True, validation_data=val_ds3)


# ### a) Model 3 - Evaluation

# In[372]:


print("Accuracy und Loss des komplexen CNN mit dataset 3:")
plot_accuracy(history3)


# Nach einigen Tests mit weiteren Layern konnte keine wesentliche Verbesserung zu Modell 2 festgestellt werden. Modell 3 zeigt ähnliche Ergebnisse und Schwächen auf wie Modell 3, trotz zusätzlicher Layer.

# In[373]:


model3_corL, model3_predL = predict_testdata(Model3, test_ds3)


# In[374]:


print(classification_report(model3_corL, model3_predL,target_names = class_names))


# Wie bei Modell 2 wurde die Klasse "owl" wieder am Schlechtesten erkannt, aus vermutlich selben Gründen.

# In[375]:


plot_confusion_matrix(confusion_matrix(model3_corL, model3_predL))


# ### e) Model 3 - Fazit

# Leider konnte in Modell 3 die Performance von Modell 2 nicht weiter optimiert werden. Auf Grund der längeren Trainingsdauer von noch komplexeren Modellen wird dies daher an dieser Stelle als Grund genommen, um nicht mehr weiter mit noch komplexeren Architekturen zu experimentieren.  
#   
# Um jedoch ein Gefühl für mögliche bessere Ergebnisse zu bekommen, sollen im nächsten Schritt vortrainierte Modelle angewendet und auf diesen Anwendungsfall optimiert werden.

# ## 6.4 Model 4 - Pretrained VGG16 Neural Network

# ### a) VGG16 - Introduction

# Das Very Deep Convolutional Networks for Large-Scale Image Recognition(VGG-16) CNN ist eines der bekanntesten vortrainierten Neuronalen Netze zur Bildklassifizierung. Es wurde mit Millionen von Bildern des ImageNet Datensetz trainiert und ist in der Lage zwischen 1000 verschiedenen Output Labels zu unterscheiden.  
# Nachfolgend ist die Architektur des Modell abgebildet. Diese setzt sich aus 13 Convolutional, 5 MaxPooling sowie 3 DenseLayern zusammen.
# 
# ![Architektur VGG16 Modell](VGG16-architecture.PNG)  
# 
# (Bildquelle1: https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/)   
# 
# Um das Modell auf einen eigenen Datensatz anwenden zu können, werden lediglich die Convolutional und MaxPooling Layer mit ihren Gewichten verwendet. An deren Output wird eine eigene Architektur aus fully-connected Layern gehängt. Wichtig ist hierbei, dass die Anzahl der Neuronen im letzten Dense Layer der Anzahl der Klassen des jeweiligen Use Cases entspricht, in unserem Projekt also 9 Neuronen für 9 verschiedene Vogelarten.

# ### b) VGG16 - Construction

# In diesem Teil wird ein Neuronales Netz erstellt, welches die Convolutional und MaxPooling Layer des VGG16 Modells verwendet. An diese wird eine eigene Zusammensetzung aus Flatten, Dense und Dropout Layern gehängt.  
#   
# Im Rahmen der Modelloptimierung wurden verschiedene Kombinationen aus Dense Layern, Dropout und Pooling Layern, sowie verschiedene Werte der Hyperparameter Image_Size, Optimizer, Learning Rate, usw. getestet. Nachfolgend wird das Modell dargestellt, welches in dieser Testphase das beste Ergebnis erzielte.  
# Für dieses Modell wurde das Dataset 2 erneut eingelesen mit image_size 128x128.

# In[100]:


#Falls ein trainiertes Modell schon existiert, kann dies hier geladen werden. Momentan auskommentiert
VGG16 = load_model('VGG16-128px-64dense.h5')


# In[102]:


#Laden des Basis Modells mit der VGG16 Architektur
baseModelVGG16 = vgg16.VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(128, 128, 3)))


# In[295]:


#Visualisierung des VGG16 Basis Modells
visualkeras.layered_view(baseModelVGG16, legend=True)


# Wie bereits in der Einleitung dieses Kapitels gezeigt, ist in der obigen Abbildung zu erkennen, dass sich das VGG16 Basis Model aus 13 Convolutional Layern und 5 MaxPooling Layern zusammensetzt.  
# Die Gewichte dieser Layer wurden zusammen mit der Architektur heruntergeladen. Diese werden nun auf trainable = False gestellt, sodass sich die Gewichte in der Trainingsphase des Modells nicht mehr verändern. In dieser sollen lediglich die Gewichte der selbst hinzugefügten Layer verändert werden.

# In[296]:


#Gewichte in den Layern des Basis Modells einfrieren, sodass diese in der Trainingsphase nicht geupdatet werden
for layer in baseModelVGG16.layers:
    layer.trainable = False


# Für das Preprocessing der Bilder bietet Keras VGG16 eine eigene Methode preprocess_input. Allerdings wurden unter Verwendung dieser Methode für train_ds und val_ds deutlich schlechtere Ergebnisse des Modells erzielt. Daher wird die Methode in diesem Notebook nicht verwendet, sondern ein eigenes Preprocessing der Bilder durchgeführt.

# In[157]:


#Preprocessing Methode von VGG16, die aber nicht verwendet wird
#for batch_images, batch_labels in train_ds:
#    batch_images = vgg16.preprocess_input(batch_images, mode='tf')


# Nun wird die Architektur des gesamten Modells bestimmt. Diesem wird zunächst ein Rescaling Layer hinzugefügt, welches das Intervall der einzelnen Bildpixel von [0,255] auf [0,1] reduziert. Anschließend wird das Basis Modell angehängt. Es folgen ein MaxPooling, Flatten, Dense und Dropout Layer bevor der finale Dense Layer mit 9 Output Neuronen angehängt wird.

# In[297]:


#Architektur des kompletten Modells
VGG16 = Sequential()
VGG16.add(Rescaling(1.0/255, input_shape=(128, 128,3)))
VGG16.add(baseModelVGG16)
VGG16.add(MaxPool2D(pool_size=(2, 2)))
VGG16.add(Flatten())
VGG16.add(Dense(64, activation="relu"))
VGG16.add(Dropout(0.5))
VGG16.add(Dense(len(class_names), activation="softmax"))
VGG16.summary()


# Nachfolgend ist die Architektur des gesamten Modells dargestellt, wobei das Basis Modell nur wie ein einzelner Layer erscheint. Dieser Layer beinhaltet die oben gezeigte Architektur.

# In[159]:


visualkeras.layered_view(VGG16, legend=True)


# ### c) VGG16 - Training

# Nun folgt das Training des Modells. Dafür wird es zunächst kompiliert und anschließend unter Verwendung der .fit() Methode über 10 Epochen trainiert. Wichtig ist hierbei nochmal, dass lediglich die Gewichte der fully-connected layer trainiert werden. Die Gewichte des VGG16 BaseModels werden nicht verändert.

# In[298]:


print("Compiling VGG16 Model")
optimizer = keras.optimizers.Adam(learning_rate=0.01, epsilon=0.1)
VGG16.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[299]:


print("Fitten des vortrainierten VGG16 Netzes mit dataset 2 (in Notebook 1 erzeugte Bilder + Bilder von https://images.cv/)")
history6 = VGG16.fit(train_ds2, batch_size=batch_size2, epochs=10, verbose=True, validation_data=val_ds2)


# Nach 10 Epochen hat das Modell eine Validation Accuracy von ca. 74%. Dies ist zwar noch ausbaufähig, übertrifft aber alle bisherigen Ergebnisse mit den selbst erstellten CNNs.  
# Auffällig ist, dass die Training Accuracy schlechter ausfällt als die Validation Accuracy. Die Erklärung hierführ könnte in dem Dropout Layer liegen, welcher zwischen die beiden Dense Layer in den fully-connected layern geschaltet wurde. Dieser deaktiviert in der Trainingsphase 50% der Neuronen, was die Accuracy in der Trainingsphase natürlich verschlechtert. Dies soll ein Overfitting des Modells verhindern. In der Validierungsphase am Ende einer Epoche werden alle Neuronen des Neuronalen Netzes aktiviert. Somit fällt die Accuracy besser aus als in der Trainingsphase.

# In[300]:


#Speichern des Modells
VGG16.save('VGG16-128px-64dense.h5')


# Nachfolgend betrachten wir die 64 Feature Maps des ersten Hidden Layer im VGG116 Modell.

# In[105]:


# definiere modell als output nach ersten hidden layer
model = Model(inputs=baseModelVGG16.inputs, outputs=baseModelVGG16.layers[1].output)
baseModelVGG16.summary()
# Bild mit richtiger Image Size laden
img = load_img(r"..\ProjectBirdClassification\Bilder2\chicken\0K3D5SDTO0RJ.jpg", target_size=(128, 128))
# Bild zu Array konvertieren
img = img_to_array(img)
# Dimension expandieren
img = expand_dims(img, axis=0)
# Peprocess Bild für VGG16
img = vgg16.preprocess_input(img)
# erzeugen der Feature Maps für ausgewählten Layer
feature_maps = model.predict(img)
# alle Feature Maps plotten
square = 8
ix = 1
plt.figure(figsize=(20,20))
for _ in range(square):
	for _ in range(square):
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
pyplot.show()


# ### d) VGG16 - Evaluation

# Nachfolgend wird das vortrainierte VGG16 Modell evaluiert. Hierfür beginnen wir mit der Betrachtung der Accuracy und des Losses über die Trainingsepochen.

# In[301]:


print("Accuracy und Loss des vortrainierten VGG16 mit dataset d:")
plot_accuracy(history6)


# Wie bereits erwähnt, ist es auffällig, dass die Werte der Valdidierungsdaten für Accuracy und Loss besser sind als für die Trainingsdaten. Eine Erklärung hierfür könnte der Dropout Layer sein.  
# In den ersten Epochen sind starke Verbesserungen in der Performance für beide Trainings- und Validationdaten zu erkennen. In den Validierungswerten sind danach einige Sprünge zu verzeichnen. Im Laufe der Epochen flachen alle Kurven immer mehr ab und pendeln sich nach ca. 8 Epochen langsam ein. Die Kurven lassen darauf schließen, dass weitere Epochen noch leichte Verbesserungen bewirken würden, eine starke Verbesserung ists aber nicht mehr anzunehmen. Daher wurde aus Laufzeitgründen auf weiteres Training verzichtet.

# Im Anschluss werden die Testdaten mit dem VGG16 Modell klassifiziert.

# In[302]:


VGG16_corL, VGG16_predL = predict_testdata(VGG16, test_ds2)


# In[303]:


print(classification_report(VGG16_corL, VGG16_predL, target_names=class_names))


# Der Classification Report zeigt die Werte der Testdaten aufgeteilt nach den verschiedenen Labeln.  
# Auffällig ist, dass die ersten drei Klassen Chicken, Cockatoo und Eagle die schlechteste Performance (Precision, Recall und F1-Score) haben. Dies ist darin zu begründen, dass in dem verwendeten Datenset 2 von diesen 3 Klassen die wenigsten Bilder vorhanden sind (siehe Kapitel zu Dataset 2). Um die Werte in den 3 Klassen zu verbessern, sind somit zusätzliche Bilder von Chicken, Cockatoo und Eagle notwendig. Diese können entweder von weiteren Quellen heruntergerladen werden oder künstlich durch Image Augmentation erzeugt werden. Die im Kapitel zu Image Augmentation gezeigten Techniken müssten hierfür für diese Klassen angewendet und die Ergebnisse physisch gespeichert und dem Datenset hinzugefügt werden.  
# Die Klasse Ostrich hat den besten Wert in Precision, die Klasse Tit für Recall und die Klasse Flamingo für F1-Score.  
# Da für unseren Use Case weder Precicion noch Recall als wichtiger erachtet werden können, schneidet im arithmetischen Mittelwert der beiden Werte (F1-Score) die Klasse Flamingo am besten ab und wird somit als die beste vom Modell vorhergesagte Klasse deklariert.

# In[304]:


plot_confusion_matrix(confusion_matrix(VGG16_corL, VGG16_predL))


# In der Confusion Matrix ist zu erkennen, dass grundsätzlich die meisten Bilder vom Modell korrekt klassifiziert wurden (diagonale Linie durch die Matrix zeigt hohe Werte). In den ersten 3 Klassen (Chicken, Cockatoo und Eagle) wurden im Vergleich zu den anderen Klassen mehr Bilder nicht korrekt klassifiziert, es gibt deutlich mehr FN und FP als in anderen Klassen. Auffällig ist hier beispielsweise, dass 42 Chicken Bilder als Cockatoo gelabeld wurden und 21 Eulen (Klasse 5) als Adler (Klasse 2). Wie bereits erwähnt, liegen für diese 3 Klassen die wenigsten Bilder vor. Um die Performance zu verbessern, bedarf es weitere Bilder dieser Klassen.  
# Ansonsten sind die Werte breit gestreut und es ist nicht zu erkennen, dass beispielsweise zwei Klassen durch das Modell häufig miteinander verwechselt wurden.

# ### e) VGG16 - Fazit

# Das vortrainierte VGG16 Modell mit eigenen fully-connected layern zeigt bisher das beste Ergebnis.  
# Zudem wurden in der Evaluation noch weitere Optimierungspotenziale durch das Hinfügen weiterer Bilder der Klassen 1-3 sowie dem Trainieren zusätzlicher Epochen erkannt. Dadurch ist anzunehmen, dass sich die Performance in allen Metriken (Gesamt-Accuracy sowie Precision, Recall und F1-Score der einzelnen Klassen) noch weiter verbessern.  
# Dieses Modell sollte somit in die finale Auswahl aufgenommen werden.

# ## 6.5 Model 5 - Pretrained Xception Neural Network

# ### a) Xception - Introduction

# Das Xception Modell ist eine Weiterentwicklung des Inception Modells. Wie das VGG16 Modell wurde auch dieses mit dem ImageNet Datenset trainiert und kann zwischen 1000 Klassen unterscheiden. Dieses Modell hat eine Tiefe von 71 Layern, ist also nochmal deutlich tiefer als das VGG16 Modell.  
# Die Architektur ist nachfolgend dargestellt:
# 
# ![Architektur Xception Modell](Xception-architecture.PNG)  
# 
# (Bildquelle2: https://medium.com/analytics-vidhya/image-recognition-using-pre-trained-xception-model-in-5-steps-96ac858f4206)
# 
# Auch bei diesem Pretrained Modell wird nur das Basismodell mit seinen Gewichten verwendet und an dessen Output eine eigene Architektur aus fully-connected layern angehängt.

# ### b) Xception - Construction

# Wie bereits beim VGG116 Modell wird auch hier nur die Basis Architektur des Xception Modells mit seinen Gewichten geladen und an diese mit einer eigenen Architektur von fully-connected layern kombiniert.  
# 
# Auch hier wurden verschiedene Architekturen von fully-connected layern, sowie Hyperparameter getestet. Es wird nachfolgend nur die Kombination gezeigt, welche hierbei das beste Ergebnis erzielte.

# In[111]:


Xcept = load_model('Xcept-256px-128-256dense.h5')


# Das Xcpetion Modell wurde ursprünglich mit einer Bildgröße von 299x299 Pixeln trainiert. Daher wird für dieses Modell eine größere Image_Size als in den vorherigen Modellen verwendet. Dafür muss zunächst das dataset 4 mit einer Image_Size von 256x256 Pixeln erstellt werden.

# In[284]:


image_size4=[256, 256]
batch_size4=128
train_ds4, val_ds4 = create_datasets(image_size4, batch_size4, r"C:\Users\olive\Documents\HdM\3. Semester\Machine Learning\Project Whale\Bilder2")
test_ds4 = val_ds4.take(25)
val_ds4 = val_ds4.skip(25)


# Anschließend wird die Basisarchitektur des Xception Modells mit seinen Gewichten geladen und diese visualisiert.

# In[106]:


baseModelXception= xception.Xception(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet')


# In[286]:


visualkeras.layered_view(baseModelXception, legend=True)


# Es wird deutlich, dass das Xception Modell eine deutlich komplexere Architektur als alle zuvor erstellten Modelle besitzt. Es wechseln sich eine Vielzahl verschiedener Dense, Pooling, Activation und Batch Layer ab.  
# Die Gewichte dieser Layer werden auch hier auf Trainable = False gestellt, sodass sie sich in der Trainingsphase nicht mehr verändern.

# In[287]:


#Gewichte in den Layern des Basis Modells einfrieren, sodass diese in der Trainingsphase nicht geupdatet werden
for layer in baseModelXception.layers:
    layer.trainable = False


# Zwar bietet auch dieses Modell eine eigene Preprocessing Methode, allerdings wurden auch hier schlechtere Ergebnisse erzielt, als wenn ein eigenes Preprocessing durchgeführt wird. Daher wird die Methode preprocess_input nicht verwendet.

# In[174]:


##for batch_images, batch_labels in train_ds:
 #   batch_images = xception.preprocess_input(batch_images)


# Da das Xception Modell ein Intervall der Bildpixel von [-1,1] erwartet, wird zunächst ein entsprechendes Rescaling durchgeführt und der Output an das Basis Modell übergeben. Anschließend folgen verschiedene fully-connected layers.

# In[288]:


Xcept = Sequential()
Xcept.add(Rescaling(scale=1./127.5, offset=-1., input_shape=(256, 256 ,3))) #Xception Model erwartet Pixelinter von [-1,1]
Xcept.add(baseModelXception)
Xcept.add(MaxPool2D(pool_size=(2, 2)))
Xcept.add(Flatten())
Xcept.add(Dense(128, activation="relu"))
Xcept.add(Dropout(0.2))
Xcept.add(Dense(256, activation="relu"))
Xcept.add(Dropout(0.5))
Xcept.add(Dense(len(class_names), activation="softmax"))
Xcept.summary()


# ### c) Xception - Training

# In[289]:


optimizer = keras.optimizers.Adam(learning_rate=0.001, epsilon=0.1)
Xcept.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[290]:


print("Fitten des vortrainierten Xception Netz mit Dataset 4")
history7 = Xcept.fit(train_ds4, batch_size=batch_size4, epochs=10, verbose=True, validation_data=val_ds4)


# Nach 10 Trainingsepochen liegen Trainings- und Validationaccuracy beide bei ca. 95%.  

# In[307]:


#Xcept.save('Xcept-256px-128-256dense.h5')


# In[110]:


# definiere modell als output nach ersten hidden layer
model = Model(inputs=baseModelXception.inputs, outputs=baseModelXception.layers[1].output)

# Bild mit richtiger Image Size laden
img = load_img(r"..\ProjectBirdClassification\Bilder2\chicken\0K3D5SDTO0RJ.jpg", target_size=(256, 256))
# Bild zu Array konvertieren
img = img_to_array(img)
# Dimension expandieren
img = expand_dims(img, axis=0)
# Peprocess Bild für VGG16
img = xception.preprocess_input(img)
# erzeugen der Feature Maps für ausgewählten Layer
feature_maps = model.predict(img)
# alle Feature Maps plotten
square=8
ix = 1
plt.figure(figsize=(20,20))
for _ in range(8):
	for _ in range(4):
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
pyplot.show()
plt.figure(figsize=(20,20))


# ### d) Xception - Evaluation

# Für die Evaluation des Modells werden zunächst der Verlauf von Accuracy und Loss über die Epochen betrachtet.

# In[308]:


print("Accuracy und Loss des vortrainierten Xception mit dataset 4 (alle Bilder und 256 Pixel):")
plot_accuracy(history7)


# In den Plots ist wieder eine starke Verbesserung von Accuracy und Loss in den ersten 3 Epochen zu erkennen. Mit der 4. Epoche werden die Verbesserungen kleiner.  
# Auffällig ist, dass die Validation Accuracy und der Loss zunächst deutlich besser als die der Trainingsdaten sind. Dies könnte wie beim VGG16 Modell an den Dropout Layern liegen. In diesem Modell verringert sich dieser Unterschied im Laufe der Epochen deutlich und ist nach 8 Epochen kaum noch vorhanden. Nach 9 Epochen sind Accuracy und Loss der Trainingsdaten siginifikant geringer als die der Validierungsdaten.  
# Es ist anzunehmen, dass das Trainieren weiterer Epochen keine deutliche Verbesserung mehr bringen würde.

# Nun werden die Testdaten durch das Modell klasifiziert und anschließend die Classification Metrics und die Confusion Matrix für diese ausgegeben.

# In[309]:


Xcept_corL, Xcept_predL = predict_testdata(Xcept, test_ds4)


# In[310]:


print(classification_report(Xcept_corL, Xcept_predL, target_names=class_names))


# Für dieses Modell liegen alle Performance Metriken (Precision, Recall und F1-Score) über 90% und schneiden somit sehr gut ab.  
# Die schlechtesten Werte verzeichnet die Klasse Owl. Auch für diese Klasse liegen vergleichsweise eher weniger Bilder vor. (siehe Kapitel zu dataset 2, hier wird zwar dataset 4 verwendet, dieses ist inhaltlich aber das dataset 2 nur mit anderen Pixelwerten). Dies kann aber nicht der einzige Grund sein, da für die Klassen 1-3 noch weniger Bilder vorhanden sind. Allerdings schneiden auch diese drei Klassen schlechter ab als die anderen. Klasse Penguin enthält ebenfalls weniger Bilder als Klasse Owl, schneidet aber trotzdem deutlich besser ab. Dass Owl trotzdem schlechter abschneidet, könnte in einer höheren Intraclass Differenz in Klasse Owl bzw. in einer geringeren Intraclass Differenz der Penguin begründet sein. Klasse Penguin weißt zudem sehr deutliche Merkmale auf (immer schwarz, weiße Vögel), weshalb diese Klasse womöglich auch mit verhältnismäßig wenig Bilder gut für das Modell erkennbar ist.
# Die Gesamtperformance des Modells lässt sich daher vermutlich noch durch das Hinzufügen weiterer Bilder dieser Klassen verbessern.  
# Den besten F1-Score hat die Klasse Ostrich, bei der Precision und Recall bei den gleichhohen Wert von 98% erzielen. 

# In[311]:


plot_confusion_matrix(confusion_matrix(Xcept_corL, Xcept_predL))


# Passend zu den Classification Metrics zeigt auch die Confusion Matrix überwiegend korrekt vorhergesagte Werte.  
# Die Klassen 1-3 sowie 5 beinhalten mehr FP und FN. Heraus sticht hierbei, dass die Klassen 2 (Eagle) und 5 (Owl) häufig verwechselt werden, da 15 Eulenbilder als Adler und 10 Adlerbilder als Eule klassifiziert wurden.  

# Wie bei Modell 2 lassen wir das Xcept Modell nun auch die separaten Testbilder klassifizieren:

# In[112]:


predict_image("..\ProjectBirdClassification\Testbilder\madagascar-penguin.jpg", [256, 256], Xcept)


# In[114]:


predict_image("..\ProjectBirdClassification\Testbilder\TucanHaribo.jpg", [256, 256], Xcept)


# In[115]:


predict_image("..\ProjectBirdClassification\Testbilder\Hedwig.jpg", [256, 256], Xcept)


# Wie Modell zwei können Pinguin und Tucan gut erkannt werden, allerdings tut sich das Xcept Modell genau wie Modell 2 mit der Eule schwer.

# ### e) Xception - Fazit

# Das Xception Modell übertrifft das aktuelle VGG16 Modell nochmal deutlich in der Gesamt-Accuracy. Auch die Precicion, Recall und F1-Score Werte der einzelnen Klasse sind deutlich besser und unterscheiden sich zudem weniger stark zwischen den Klassen.

# # 7. Projekt Fazit

# Bei den selbst erstellten Modellen konnte das Ziel von 80% Accuracy nicht erreicht werden. Durch Tests und Optimierungen mit verschiedenen Architekturen und Parametern konnte eine Accuracy von knapp 70% erreicht werden. Durch das hinzufügen weiterer Trainingsdaten, vor allem für weniger stark vertretene Klassen, könnte dieser Wert vermutlich noch optimiert werden.  
# 
# Bei den vortrainierten Modellen konnte wie erwartet ein deutlich besseres Ergebnis erzielt werden. Das Xception Modell lieferte mit einer Accuracy von ca. 95% das deutlich beste Ergebnis und wäre somit auch für den praktischen Einsatz geeignet. Auch die vortrainierten Netze könnten vermutlich mit mehr Ressourcen noch leicht optimiert werden.

# # 8. Quellen

# zusätzliche Bilder: https://images.cv/  
# Literatur zu Dropout: http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf  
# Bildquelle 1: https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/  
# Bildquelle 2: https://medium.com/analytics-vidhya/image-recognition-using-pre-trained-xception-model-in-5-steps-96ac858f4206

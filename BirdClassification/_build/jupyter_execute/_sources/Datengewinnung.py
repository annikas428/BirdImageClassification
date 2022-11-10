#!/usr/bin/env python
# coding: utf-8

# # Notebook 1 - Datengewinnung
# von Annika Scheug und Oliver Schabe

# Da es Im Internet bereits eine Vielzahl an Bilderdatensets und Bilderdatenbanken mit darauf aufbauenden Lösungen gibt, haben wir uns dazu entschieden, unser eigenes Datenset für eine Bilderklassifikation zu erstellen.  
# Dazu nutzen wir Selenium, ein open source Tool zur Automatisierung von Browseroperationen.  
# Mit Hilfe von Selenium durchsuchen wir Google Bilder automatisiert nach bestimmten Suchbegriffen und speichern die gefunden Bildern lokal auf unserem Rechner ab.  
#   
# Auf Basis des so generierten Datensets soll dann ein Neuronales Netz zur Erkennung von Bildern trainiert werden. Als zu klassifizierende Objekte haben wir uns für verschiedene Vogelarten entschieden.  
# 
# Die Interclass Varianz soll dabei nicht zu hoch sein, um eine angemessene Schwierigkeit sicherzustellen (da alles Vögel sind und in der Regel Schnabel, Flügel etc. haben). Um ein gutes Modell zu entwicklen, sollte die Interclass Varianz allrdings auch nicht zu niedrig sein, da sonst die richtige Klassifizierung sehr schwierig werden könnte. 
# 
# Auf Basis dieser Überlegung haben wir uns für folgende 9 Vogelarten entschieden, welche viele Merkmale wie Flügel und Schnäbel teilen, sich allerdings optisch stark genug voneinander unterscheiden um im Rahmen dieses Projektes ein gut funktionierendes Modell zu entwicklen:
# * eagle (Adler)  
# *tit (Meise)  
# *owl (Eule)  
# *tucan (Tukan)  
# *flamingo (Flamingo)    
# *ostrich (Vogelstrauß)  
# *cockatoo (Kakadu)  
# *chicken (Huhn) 
# *penguin (Pinguin) 
# 
# Die Intraclass Varianz unterscheidet sich dabei bei den verschiedenen Klassen teilweise stark. Bei Adlern bspw. gibt es viel mehr Unterarten, die sich auch optisch voneinander Unterscheiden als bei Flamingos. Dies wird sich voraussichtlich in den Ergebnissen des Modells wiederspiegeln. Vermutlich werden Klassen mit höherer Intraclass Varianz schlechtere Ergebnisse erzielen.  
#   
# ![AdlerVarianz](AdlerVarianz.png)

# ## Setup

# Zuerst werden die benötigten packages und libraries installiert bzw importiert.

# In[1]:


# Installieren von Selenium
#pip install selenium


# In[2]:


#Importieren von Selenium Libraries
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time


# ## Erzeugung des Datasets

# Hier wird zunächst eine Liste erstellt, welche alle Suchbegriffe enthält, zu denen Bilder gesucht werden sollen.

# In[3]:


# List erstellen von zu klassifizierenden Vögeln
Labels = ["eagle","tit","owl","tucan","flamingo","ostrich","cockatoo","chicken","penguin"]
Labels


# Der folgende Code durchsucht mit Hilfe von Selenium Google Bilder nach den definierten Suchbegriffen und speichert die gefundenen Bilder lokal ab. Die automatisierte Navigation durch den Chrome Browser erfolgt auf Basis des "xpath".  
# Der xpath gibt die genaue Platzierung eines Objekts (bspw. Google Suche Eingabefeld) im html Quellcode an. 
#   
# Details zur Methode sind im Code direkt kommentiert.

# In[4]:


# Definition einer Funktion zum Durchscrollen der Google Bilder Suchergebnisse
def scroll_to_bottom():

	last_height = driver.execute_script('	return document.body.scrollHeight')

	while True:
		driver.execute_script('		window.scrollTo(0,document.body.scrollHeight)')

		# kurz warten um Ergebnisse laden zu lassen
		time.sleep(3)

		new_height = driver.execute_script('		return document.body.scrollHeight')

		# klicken von "weitere Ergebnisse anzeigen"
		try:
			driver.find_element("xpath", "/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div[1]/div[2]/div[2]/input").click()

			# kurz warten um Ergebnisse laden zu lassen
			time.sleep(1.5)

		except:
			pass

		# Prüfen ob das Ende der Seite erreicht wurde
		if new_height == last_height:
			break

		last_height = new_height


# In[5]:


# Schleife zum automatiserten Suchen und Downloaden von Google Bilder Suchergebnissen zu den definierten Klassen:
for birds in Labels:

	# webdriver instance erstellen
	driver = webdriver.Chrome(r'..\ProjectBirdClassification\chromedriver1.exe')
	driver.implicitly_wait(3)
	# Browser Fenster maximieren
	driver.maximize_window()


	# Google Images öffnen
	driver.get('https://images.google.com/')


	#Cookie Window akzeptieren
	driver.find_element("xpath", '/html/body/div[2]/div[2]/div[3]/span/div/div/div/div[3]/div[1]/button[2]/div').click()


	# Suchleiste auswählen
	box = driver.find_element("xpath", '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')

	# Suchanfrage eingeben (Vogelarten). Um die Qualität der Suchergebnisse zu erhöhen,  
    # wird außerdem zum Name des Vogels noch der string "bird photo" ergänzt
	box.send_keys(birds + ' bird photo')

	# Suchanfrage abschicken
	box.send_keys(Keys.ENTER)

	# Aufrufen der zuvor definierten Scroll Funktion, um mehr als nur die ersten Suchergebniss runterladen zu können
	scroll_to_bottom()


	# Schleife um die ersten 600 gefunden Bilder ja Klasse zu speichern
	for i in range(1, 1000):

		try:
			try:
				# XPath der Bilder, dieser ändert sich nur an einer Stelle und zählt von 1 aufsteigend durch  
                # (Erstes Suchergebnis hat die 1, zweites die 2 usw. an der entsprechenden Stelle im Pfad)
				print("//*[@id='islrg']/div[1]/div[" +
					str(i) + "]/a[1]/div[1]/img")
				img = driver.find_element("xpath",
					'//*[@id="islrg"]/div[1]/div[' +
					str(i) + ']/a[1]/div[1]/img')

			except:
				print("Element konnte nicht gefunden werden")
			# Pfad, unter dem die Bilder lokal abgespeichert werden sollen
			img.screenshot(r'..\ProjectBirdClassification\Bilder\\' +
						birds + ' (' + str(i) + ').png')

			# kurz warten um Error zu vermeiden
			time.sleep(0.2)
		except:
			print("mistake")
			# Wenn Bild nicht gefunden und gespeichert werden kann, gehen wir weiter zum nächsten Bild
			continue
            
	# Schließen des Browsers
	driver.close()


# Mit der hier verwendeten Methode können auch einfach für andere Klassifikationsaufgaben Bilder gewonnen werden um Modelle zur Erkennung von anderen Objekten zu trainieren. Es müssen lediglich die Suchbegriffe in der oben definierten Liste ausgetauscht werden.
# 
# Nachteil der verwendeten Methode ist die teilweise durchmischte Qualität der Suchergebnisse. Immer wieder tauchen Bilder unter dem Suchbegriff auf, welche nicht das gesuchte Objekt abbilden oder in sehr abgeänderter/abstrakter Form. Diese müssen dann manuell entfernt werden, um eine hohe Qualität das Datensets sicherzsutellen.  
# 
# Die hier gewonnen Bilder sollen nun in einem nächste Notebook zum Trainieren eines CNN verwendet werden.

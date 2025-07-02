# Graph Neural Networks voor robuuste cijferreconstructie en automatische uitlezing van analoge meters op embedded hardware

Dit project bevat mijn afstudeeronderzoek voor de master applied AI. Hierin vind je mijn volledige scriptie en de bijbehorende code die ik heb gebruikt voor analyses en experimenten.

## Inhoud van deze repository
- **Scriptie_Lenka_Piet_MAAI.pdf**  
  De volledige scriptie met achtergrond, methoden, resultaten en conclusies van mijn onderzoek.

- **code/**  
  Map met Python-scripts en Jupyter Notebooks die gebruikt zijn voor de data-analyse en experimenten.

  - **code_volledige_pipeline/**  
    Bevat de volledige pipeline als Python bestand. Deze kan zelf eenvoudig worden uitgevoerd. Zowel de pipeline met en zonder reconstuctie is aanwezig.

  - **modellen_evalueren/**  
    Scripts en notebooks gericht op het evalueren van verschillende modellen en de volledige pipeline, inclusief prestatiebeoordelingen en vergelijkingen.

  - **modellen_trainen/**  
    Code voor het trainen van de modellen. Het volledig iteratief proces is hierin opgenomen. Voor uitgebreide beargumentatie wordt verwezen naar de scriptie.

  - **syntetische_data_maken/**  
    Scripts voor het genereren van synthetische datasets ter ondersteuning van modeltraining en -evaluatie.

## Over het onderzoek

Dit project richt zich op de ontwikkeling van een robuust en energiezuinig Automatic Meter Reading (AMR)-systeem voor analoge meters in industriële omgevingen. Het systeem maakt gebruik van beeldherkenning om meterstanden automatisch uit te lezen, zelfs onder uitdagende omstandigheden zoals slechte verlichting, vervuiling en beschadigingen.

Het AMR-systeem bestaat uit drie onderdelen:

- Een detectiemodel (YOLOv11nano) voor het lokaliseren van het cijferdisplay.
- Een compact neuraal netwerk voor het classificeren van cijfers.
- Een reconstructiemodel gebaseerd op een Graph Neural Network (GNN) voor het herstellen van vervormde of deels verborgen cijfers.

Door gebruik te maken van TinyML-technieken is het systeem geoptimaliseerd voor resource-beperkte hardware zoals microcontrollers, waardoor lokale, energiezuinige verwerking mogelijk is zonder netwerkverbinding.

Let op: het GNN-reconstructiemodel is op dit moment nog niet geschikt voor embedded hardware en vormt een aandachtspunt voor verdere ontwikkeling.

Hoewel het systeem onder gecontroleerde omstandigheden goed presteert, is de robuustheid in realistische, industriële situaties nog beperkt.
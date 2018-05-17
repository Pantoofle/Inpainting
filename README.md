Utilisation du programme :
- `img/` : les images
- `src/` : le programme, les sources
- `CImg` : la librairie externe

Dans `img/`
- Plusieurs images, sous format .jpg
- Chaque image.jpg doit être accompagnée par image_msk.jpg, son masque
- Le masque est blanc pour les zones à conserver, noir pour les zones à remplire
- Le programme se lance sur input.jpg et msk.jpg. Le résultat est écrit au fil de l'éxecution du programme dans res.jpg
- Pour copier les bons fichiers et masques au bon endroit, utiliser le script `./setup.sh <image_sans_extension>`.
	Ce script copie l'image dans input.jpg et le masque dans msk.jpg
- Le dossier examples contient quelque exemples d'input/mask/output de l'algorithme

Dans src/
- Taper `make` pour compiler le programme. 
- Les librairies nécessaires sont `-lm -lpthread -lX11`. Le reste est inclut automatiquement dans la lib CImg qui est dans son dossier
- `make run` lance le programme sur l'input avec une taille de patch par défaut de 11
- Pour entrer soi même la taille du patch, entrer `make run SIZE=<taille>`. ATTENTION, cette taille doit être impaire, pour centrer le patch sur un pixel


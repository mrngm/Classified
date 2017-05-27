# Classified

* [Dataset](https://surfdrive.surf.nl/files/index.php/s/K2FYXiWVb8B9yMH),
* [Git crash course](https://blyt.es/gitcc.pdf)
    * Hierin staan wat concepten van Git uitgelegd. Mocht je vragen hierover
      hebben, laat maar weten.

## Installatie xgboost

```
$ source init.sh        # zorg ervoor dat Anaconda wordt gebruikt
$ git clone --recursive https://github.com/dmlc/xgboost
$ cd xgboost/
$ git checkout v0.60
$ git submodule init
$ git submodule update
### If you're running Windows, make sure mingw32-make and g++ are in your PATH
### e.g.
### $ which mingw32-make
### <expected output: the location of mingw32-make>
### $ alias make='mingw32-make'
### $ which g++
### <expected output: the location of g++>
### If running `make -j4` after that doesn't immediately work, make sure
### that dmlc-core/ and rabit/ are non-empty, otherwise run `git submodule init`
### and `git submodule update` again. After that:
### $ cd dmlc-core
### $ make -j4
### $ cd ../rabit/
### $ make lib/librabit_empty.a -j4
### $ cd ../
### $ cp make/mingw64.mk config.mk
$ make -j20
$ cd python-package
$ python -V
Python 2.7.13 :: Anaconda 4.3.1 (64-bit)
$ python setup.py install
$ conda install libgcc
$ ipython
In [1]: import xgboost
[bla]: DeprecationWarning: [bla]

In [2]:
```

[Instructie voor bouwen van XGboost onder Windows](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en)

## Committen e.d.

Er zijn meerdere manieren om dingen aan deze repository toe te voegen. De ene
werkt met zogenaamde *fork*s, de ander is met branches op deze repository.

### Branches

1. Zorg ervoor dat je lokale *clone* up-to-date is. Dit kan je vanaf de
commandline doen met `git pull`. Met iets als [TortoiseGit](https://tortoisegit.org/)
kan je dat met knopjes doen, direct vanuit iets als Windows Explorer, Nautilus,
Finder.app, of wat dies meer zij. Zie [de pagina met screenshots](https://tortoisegit.org/about/screenshots/)
voor wat visuele hints.
    * Je kan ook je lokale kopie weggooien en deze repository opnieuw clonen.
2. Controleer nadat je hebt ge*pull*d dat je op de *master* branch zit.
    * `git status`
    * of: `git branch` (de huidige branch staat aangemerkt met een `*`)
3. Maak een branch aan met `git checkout -b branchnaam`
    * Vervang `branchnaam` met een naam voor je branch, bijvoorbeeld
      `gerdriaan-scripts`
4. Doe je reguliere commits
    * Vergeet de "zinnige" commit-berichten niet
    * "Bestanden bijgewerkt" is een voorbeeld van een niet-zinnig commitbericht.
      Het idee is dat je aan het commit-bericht kan zien wat er zoal is gebeurd.
    * Heb je veel dingen gedaan? Splits dan je werk op in meerdere commits. Op
      de commandline kan dat met `git add -p`. Je kan dan per stukje code (of
      tekst, of ...) bepalen of het in de commit terecht moet komen.
    * Bestanden verplaatsen kan met `git mv bron doel`, bijvoorbeeld: `git mv
      "Notities 23 Mei - Voorbespreking.txt" Notities/`
5. *Push* je commits naar Github
    * `git push origin branchnaam`, bijvoorbeeld: `git push origin gerdriaan-scripts`
    * `origin` is de naam waaronder Github bekend staat in je lokale kopie
6. Maak een [*pull request*](https://github.com/RoelBouman/Classified/pull/new/master)
    * Als "base" gebruik je `master`, als "compare" gebruik je `branchnaam`, dus
      bijvoorbeeld `gerdriaan-scripts`
    * Beschrijf wat je wil toevoegen
    * Klik op "Create pull request"

Typisch gezien maak je voor elk nieuw stukje wat je wil toevoegen aan `master`
een nieuwe branch. Stel ik zou twee pipelines maken, dan zou ik op een gegeven
moment vanaf `master` afsplitsen (stap 3 hierboven). Die branch zou dan
`gerdriaan-pipeline1` kunnen heten.

Als er ondertussen updates zijn geweest aan `master` (iemand anders heeft een
stukje preprocessing toegevoegd), dan zorg ik ervoor dat in mijn lokale kopie,
`master` weer up-to-date is (zie stap 1). Daarna volgt uiteraard stap 3 en de
rest.

Om de zoveel tijd kan het handig zijn om de *remote* branches die nog lokaal
staan te verwijderen. Dat kan met `git remote prune origin`. Branches die je
zelf hebt aangemaakt kan je verwijderen met `git branch -d <branchnaam>`. Je
kan hier een melding krijgen dat deze branch nog niet gemerged is. Mocht je
zeker weten dat dit wel gebeurd is (of het boeit je niet) dan kan je met
`git branch -D <branchnaam>` 'm alsnog verwijderen.

### Forks

Forks werken vergelijkbaar met branches. Het verschil is dat je over je fork
volledige rechten hebt.

Je moet het volgende in het achterhoofd houden:
* je fork wordt niet automatisch geupdatetetet
* je moet zelf ervoor zorgen dat jouw fork up to date is

Typisch gezien, als je een fork hebt gemaakt *clone* je jouw fork en maak je een
nieuwe "remote" aan, zodat je van de originele repository de laatste commits
binnen kan halen:
* `git clone git@github.com:mrngm/Classified`
* `git remote add upstream git@github.com:RoelBouman/Classified`

Als je dan op jouw fork de `master` branch wil updaten voer je dit uit op jouw
lokale kopie van jouw fork:
* `git checkout master`
* `git pull upstream master`

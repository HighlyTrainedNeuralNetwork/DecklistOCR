# DecklistOCR

This is a project for generating MTG decklists from images. 
Tangrams did this first (https://github.com/davidcinglis/netdecker). I saw some issues with that and wanted to try reimplementing it.
I did not want to deal with a larger library just to get DBSCAN so I am using https://github.com/chrisjmccormick/dbscan for that.

## TODO
- This logic seems to be overfitted on my MTGO training images and tanks on Arena ones. That needs to be fixed.
- I'm making a frontend for this. To that end lambdaImplementation.py needs to not be blank.
- Cut out numpy dependency and investigate cutting out others.
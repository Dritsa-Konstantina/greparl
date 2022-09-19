# A Greek Parliament Proceedings Dataset for Computational Linguistics and Political Analysis

This repository includes the source code for the collection, cleaning and computational study of the [Greek Parliament Proceedings Dataset](https://zenodo.org/record/7005201). Detailed instructions and descriptions of all the scripts are included in the supplementary material.

The dataset includes 1,280,918 speeches (rows) of Greek parliament members with a total volume of 2.12 GB, that were exported from 5,355 parliamentary sitting record files. They extend chronologically from early July 1989 up to late July 2020. The dataset consists of a .csv file and includes the following columns of data:
- member_name: the official name of the parliament member who talked during a sitting.
- sitting_date: the date that the sitting took place.
- parliamentary_period: the name and/or number of the parliamentary period that the speech took place in. A parliamentary period includes multiple parliamentary sessions.
- parliamentary_session: the name and/or number of the parliamentary session that the speech took place in. A parliamentary session includes multiple parliamentary sittings.
- parliamentary_sitting: the name and/or number of the parliamentary sitting that the speech took place in.
- political_party: the political party that the speaker belonged to the moment of their speech.
- government: the government in force when the speech took place.
- member_region: the electoral district the speaker belonged to.
- roles: information about the parliamentary roles and/or government position of the speaker the moment of their speech.
- member_gender: the sex of the speaker
- speech: the speech that the member made during the parliamentary sitting

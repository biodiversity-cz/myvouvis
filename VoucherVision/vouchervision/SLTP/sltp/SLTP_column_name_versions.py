class ColumnVersions:
    
    def __init__(self):

        # Class property for SLTPvA columns
        self.taxonomy_columns = ['family','genus','specificEpithet','infraspecificEpithet','scientificName','scientificNameAuthorship', 'verbatimTaxonRank',]
        self.taxonomy_json = self.list_to_json(self.taxonomy_columns)



        self.wcvp_columns = ['family','genus','specificepithet','infraspecificepithet','scientfiicname','scientfiicnameauthorship','taxonrank',]
        self.wcvp_json = self.list_to_json(self.wcvp_columns)
        


        self.MICH_to_SLTPvA_columns = ['catalogNumber', 'order', 'family', 'scientificName', 'scientificNameAuthorship', 'genus', 'subgenus', 'specificEpithet', 'infraspecificEpithet', 'identifiedBy', 'recordedBy', 'recordNumber', 'verbatimEventDate', 'eventDate', 'habitat', 'occurrenceRemarks', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'degreeOfEstablishment','decimalLatitude', 'decimalLongitude', 'verbatimCoordinates', 'minimumElevationInMeters', 'maximumElevationInMeters',]
        self.MICH_to_SLTPvA_json = self.list_to_json(self.MICH_to_SLTPvA_columns)


        self.conversion = {'degreeOfEstablishment':'establishmentMeans',}


        self.MICH_to_SLTPvA_columns_Morton_translation = {'SpecimenBarcode':'catalogNumber', 
                                                          '':'order', 
                                                          'FamilyName':'family', 'CalcFullName':'scientificName', 'CalcFullName':'scientificNameAuthorship', 'GenusName':'genus', 
                                                          'Subgenus':'subgenus', 'SpeciesName':'specificEpithet', 'Subspecies':'infraspecificEpithet', 
                                                          '':'identifiedBy', 
                                                          'Collectors':'recordedBy', 'FieldNumber':'recordNumber', 'DateText':'verbatimEventDate', 
                                                          '':'eventDate', '':'habitat', '':'occurrenceRemarks', '':'country', '':'stateProvince', 
                                                          '':'county', '':'municipality', '':'locality', '':'degreeOfEstablishment',
                                                          '':'decimalLatitude','':'decimalLongitude', '':'verbatimCoordinates', 
                                                          'Elevation':'minimumElevationInMeters', 'ElevationMax':'maximumElevationInMeters'}

        
        
        # MICH_to_SLTPvA_json = {
        #     "catalogNumber": "",
        #     "order": "",
        #     "family": "",
        #     "scientificName": "",
        #     "scientificNameAuthorship": "",
        #     "genus": "",
        #     "subgenus": "",
        #     "specificEpithet": "",
        #     "verbatimTaxonRank": "",
        #     "infraspecificEpithet": "",
        #     "identifiedBy": "",
        #     "recordedBy": "",
        #     "recordNumber": "",
        #     "verbatimEventDate": "",
        #     "habitat": "",
        #     "occurrenceRemarks": "",
        #     "associatedTaxa": "",
        #     "country": "",
        #     "stateProvince": "",
        #     "county": "",
        #     "municipality": "",
        #     "locality": "",
        #     "decimalLatitude": "",
        #     "decimalLongitude": "",
        #     "verbatimCoordinates": "",
        #     "minimumElevationInMeters": "",
        #     "maximumElevationInMeters": "",
        #     }
        
        # taxonomy_json = {
        #     "family": "",
        #     "genus": "",
        #     "specificEpithet": "",
        #     "infraspecificEpithet": "",
        #     "scientificName": "",
        #     "scientificNameAuthorship": "",
        #     "verbatimTaxonRank": "",
        #     }
    
    # Methods to return properties
    def get_taxonomy_columns(self):
        return self.taxonomy_columns

    def get_taxonomy_json(self):
        return self.taxonomy_json

    def get_wcvp_columns(self):
        return self.wcvp_columns

    def get_wcvp_json(self):
        return self.wcvp_json

    def get_MICH_to_SLTPvA_columns(self):
        return self.MICH_to_SLTPvA_columns

    def get_MICH_to_SLTPvA_json(self):
        return self.MICH_to_SLTPvA_json
    
    def get_conversions(self):
        return self.conversion


    @staticmethod
    def list_to_json(column_list):
        return {column: "" for column in column_list}
    
    
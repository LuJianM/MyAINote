# MyAINote







Abstract—Given the large variety and complexityoftables,tablestructureextractionisachallengingtaskinautomateddocumentanalysissystems.Wepresentapairofnoveldeeplearningmodels(SplitandMergemodels)thatgivenaninputimage,1)predictsthebasictablegridpatternand2)predictswhichgridelementsshouldbemergedtorecovercellsthatspanmultiplerowsorcolumns.WeproposeprojectionpoolingasanovelcomponentoftheSplitmodelandgridpoolingasanovelpartoftheMergemodel.WhilemostFullyConvolutionalNetworksrelyonlocalevidence,theseuniquepoolingregionsallowourmodelstotakeadvantageoftheglobaltablestructure.Weachievestate-of-the-artperformanceonthepublicICDAR2013TableCompetitiondatasetofPDFdocuments.Onamuchlargerprivatedatasetwhichweusedtotrainthemodels,wesigniﬁcantlyoutperformbothastate-of-
the-artdeepmodelandamajorcommercialsoftwaresystem.
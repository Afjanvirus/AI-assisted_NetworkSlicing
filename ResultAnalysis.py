import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# --- 1. Raw Log Data ---
# All log entries are included.
LOG_DATA = """
2025-07-30 09:42:30.205: Got KPIs from Qosium: [{"downlink": {"throughput": 3912.0, "latency": 6.5322456, "jitter": 1.421, "packetLoss": 0.0}, "uplink": {"throughput": 6904.0, "latency": 8.167754, "jitter": 1.615, "packetLoss": 0.0}}, {"downlink": {"throughput": 3744.0, "latency": 5.364, "jitter": 0.829, "packetLoss": 0.0}, "uplink": {"throughput": 6080.0, "latency": 9.551, "jitter": 2.214, "packetLoss": 0.0}}]
2025-07-30 09:42:30.217: Got DL slice allocation; Slice1=0.5647, Slice2=0.4353
2025-07-30 09:42:30.217: Got UL slice allocation; Slice1=0.6029, Slice2=0.3971
2025-07-30 09:42:32.253: Got KPIs from Qosium: [{"downlink": {"throughput": 3112.0, "latency": 6.0927815, "jitter": 0.59, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 7.8872185, "jitter": 0.722, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 6.418, "jitter": 3.464, "packetLoss": 0.0}, "uplink": {"throughput": 4032.0, "latency": 10.658, "jitter": 4.623, "packetLoss": 0.0}}]
2025-07-30 09:42:32.263: Got DL slice allocation; Slice1=0.5778, Slice2=0.4222
2025-07-30 09:42:32.263: Got UL slice allocation; Slice1=0.6146, Slice2=0.3854
2025-07-30 09:42:34.290: Got KPIs from Qosium: [{"downlink": {"throughput": 3304.0, "latency": 7.638433, "jitter": 1.615, "packetLoss": 0.0}, "uplink": {"throughput": 5880.0, "latency": 8.061567, "jitter": 0.72, "packetLoss": 0.0}}, {"downlink": {"throughput": 3112.0, "latency": 4.593, "jitter": 0.604, "packetLoss": 0.0}, "uplink": {"throughput": 4032.0, "latency": 8.217, "jitter": 1.323, "packetLoss": 0.0}}]
2025-07-30 09:42:34.300: Got DL slice allocation; Slice1=0.613, Slice2=0.387
2025-07-30 09:42:34.300: Got UL slice allocation; Slice1=0.6471, Slice2=0.3529
2025-07-30 09:42:36.337: Got KPIs from Qosium: [{"downlink": {"throughput": 2696.0, "latency": 6.021, "jitter": 0.296, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 7.516, "jitter": 1.157, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.638309, "jitter": 0.062, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.077691, "jitter": 2.58, "packetLoss": 0.0}}]
2025-07-30 09:42:36.346: Got DL slice allocation; Slice1=0.6386, Slice2=0.3614
2025-07-30 09:42:36.346: Got UL slice allocation; Slice1=0.6678, Slice2=0.3322
2025-07-30 09:42:38.378: Got KPIs from Qosium: [{"downlink": {"throughput": 3912.0, "latency": 6.410539, "jitter": 0.197, "packetLoss": 0.0}, "uplink": {"throughput": 6904.0, "latency": 8.182461, "jitter": 1.42, "packetLoss": 0.0}}, {"downlink": {"throughput": 3136.0, "latency": 5.272842, "jitter": 0.526, "packetLoss": 0.0}, "uplink": {"throughput": 5928.0, "latency": 9.935158, "jitter": 3.377, "packetLoss": 0.0}}]
2025-07-30 09:42:38.387: Got DL slice allocation; Slice1=0.6002, Slice2=0.3998
2025-07-30 09:42:38.387: Got UL slice allocation; Slice1=0.6349, Slice2=0.3651
2025-07-30 09:42:40.418: Got KPIs from Qosium: [{"downlink": {"throughput": 2696.0, "latency": 5.815960884094238, "jitter": 0.5550000071525574, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 8.191039085388184, "jitter": 1.1069999933242798, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.698999881744385, "jitter": 0.7749999761581421, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.541000366210938, "jitter": 1.5210000276565552, "packetLoss": 0.0}}]
2025-07-30 09:42:40.426: Got DL slice allocation; Slice1=0.5272, Slice2=0.4728
2025-07-30 09:42:40.426: Got UL slice allocation; Slice1=0.5683, Slice2=0.4317
2025-07-30 09:42:42.464: Got KPIs from Qosium: [{"downlink": {"throughput": 119240.0, "latency": 7.97599983215332, "jitter": 1.6540000438690186, "packetLoss": 0.0}, "uplink": {"throughput": 32135232.0, "latency": 44.10200119018555, "jitter": 3.509999990463257, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 5.15541934967041, "jitter": 1.1059999465942383, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.988580703735352, "jitter": 2.484999895095825, "packetLoss": 0.0}}]
2025-07-30 09:42:42.472: Got DL slice allocation; Slice1=0.7947, Slice2=0.2053
2025-07-30 09:42:42.472: Got UL slice allocation; Slice1=0.7901, Slice2=0.2099
2025-07-30 09:42:44.504: Got KPIs from Qosium: [{"downlink": {"throughput": 128328.0, "latency": 8.222000122070312, "jitter": 1.4359999895095825, "packetLoss": 0.0}, "uplink": {"throughput": 65848016.0, "latency": 10.892999649047852, "jitter": 0.30799999833106995, "packetLoss": 0.0}}, {"downlink": {"throughput": 3720.0, "latency": 4.960790157318115, "jitter": 0.7770000100135803, "packetLoss": 0.0}, "uplink": {"throughput": 5056.0, "latency": 8.699210166931152, "jitter": 1.9579999446868896, "packetLoss": 0.0}}]
2025-07-30 09:42:44.511: Got DL slice allocation; Slice1=0.7948, Slice2=0.2052
2025-07-30 09:42:44.511: Got UL slice allocation; Slice1=0.7913, Slice2=0.2087
2025-07-30 09:42:46.542: Got KPIs from Qosium: [{"downlink": {"throughput": 137672.0, "latency": 9.583000183105469, "jitter": 1.152999997138977, "packetLoss": 0.0}, "uplink": {"throughput": 90549440.0, "latency": 11.39799976348877, "jitter": 0.25, "packetLoss": 0.0}}, {"downlink": {"throughput": 3136.0, "latency": 5.216000080108643, "jitter": 1.2339999675750732, "packetLoss": 0.0}, "uplink": {"throughput": 6800.0, "latency": 9.293000221252441, "jitter": 1.5820000171661377, "packetLoss": 0.0}}]
2025-07-30 09:42:46.550: Got DL slice allocation; Slice1=0.7944, Slice2=0.2056
2025-07-30 09:42:46.550: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-30 09:42:48.588: Got KPIs from Qosium: [{"downlink": {"throughput": 125192.0, "latency": 8.041000366210938, "jitter": 2.431999921798706, "packetLoss": 0.0}, "uplink": {"throughput": 110440584.0, "latency": 13.975000381469727, "jitter": 0.8050000071525574, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.575864315032959, "jitter": 0.12800000607967377, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.484135627746582, "jitter": 5.452000141143799, "packetLoss": 0.0}}]
2025-07-30 09:42:48.595: Got DL slice allocation; Slice1=0.7945, Slice2=0.2055
2025-07-30 09:42:48.595: Got UL slice allocation; Slice1=0.7904, Slice2=0.2096
2025-07-30 09:42:50.643: Got KPIs from Qosium: [{"downlink": {"throughput": 119560.0, "latency": 9.416000366210938, "jitter": 4.203999996185303, "packetLoss": 0.0}, "uplink": {"throughput": 105042248.0, "latency": 10.680999755859375, "jitter": 0.4449999928474426, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.723133563995361, "jitter": 0.04500000178813934, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.941865921020508, "jitter": 3.2290000915527344, "packetLoss": 0.0}}]
2025-07-30 09:42:50.652: Got DL slice allocation; Slice1=0.7952, Slice2=0.2048
2025-07-30 09:42:50.652: Got UL slice allocation; Slice1=0.7917, Slice2=0.2083
2025-07-30 09:42:52.683: Got KPIs from Qosium: [{"downlink": {"throughput": 99816.0, "latency": 9.970999717712402, "jitter": 3.13700008392334, "packetLoss": 0.0}, "uplink": {"throughput": 70350560.0, "latency": 11.744999885559082, "jitter": 0.5270000100135803, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.630978107452393, "jitter": 0.09700000286102295, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.053021430969238, "jitter": 4.673999786376953, "packetLoss": 0.0}}]
2025-07-30 09:42:52.692: Got DL slice allocation; Slice1=0.7947, Slice2=0.2053
2025-07-30 09:42:52.692: Got UL slice allocation; Slice1=0.791, Slice2=0.209
2025-07-30 09:42:54.723: Got KPIs from Qosium: [{"downlink": {"throughput": 119368.0, "latency": 7.14900016784668, "jitter": 1.034000039100647, "packetLoss": 0.0}, "uplink": {"throughput": 97469968.0, "latency": 11.420999526977539, "jitter": 0.23800000548362732, "packetLoss": 0.0}}, {"downlink": {"throughput": 3136.0, "latency": 4.849826335906982, "jitter": 0.8389999866485596, "packetLoss": 0.0}, "uplink": {"throughput": 5928.0, "latency": 9.963173866271973, "jitter": 2.1640000343322754, "packetLoss": 0.0}}]
2025-07-30 09:42:54.732: Got DL slice allocation; Slice1=0.7945, Slice2=0.2055
2025-07-30 09:42:54.732: Got UL slice allocation; Slice1=0.7907, Slice2=0.2093
2025-07-30 09:42:56.765: Got KPIs from Qosium: [{"downlink": {"throughput": 125960.0, "latency": 9.883999824523926, "jitter": 2.822999954223633, "packetLoss": 0.0}, "uplink": {"throughput": 111670016.0, "latency": 12.03600025177002, "jitter": 0.6809999942779541, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.4029998779296875, "jitter": 0.7409999966621399, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.12600040435791, "jitter": 1.2940000295639038, "packetLoss": 0.0}}]
2025-07-30 09:42:56.774: Got DL slice allocation; Slice1=0.7953, Slice2=0.2047
2025-07-30 09:42:56.774: Got UL slice allocation; Slice1=0.7918, Slice2=0.2082
2025-07-30 09:42:58.810: Got KPIs from Qosium: [{"downlink": {"throughput": 120616.0, "latency": 8.99899959564209, "jitter": 3.1489999294281006, "packetLoss": 0.0}, "uplink": {"throughput": 87339904.0, "latency": 10.020999908447266, "jitter": 0.9290000200271606, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.8429999351501465, "jitter": 1.347000002861023, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.154999732971191, "jitter": 1.6360000371932983, "packetLoss": 0.0}}]
2025-07-30 09:42:58.819: Got DL slice allocation; Slice1=0.7953, Slice2=0.2047
2025-07-30 09:42:58.819: Got UL slice allocation; Slice1=0.7921, Slice2=0.2079
2025-07-30 09:43:00.854: Got KPIs from Qosium: [{"downlink": {"throughput": 132264.0, "latency": 10.480999946594238, "jitter": 1.8309999704360962, "packetLoss": 0.0}, "uplink": {"throughput": 110624512.0, "latency": 12.8100004196167, "jitter": 0.24199999868869781, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.7789998054504395, "jitter": 1.184999942779541, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.875, "jitter": 1.9589999914169312, "packetLoss": 0.0}}]
2025-07-30 09:43:00.862: Got DL slice allocation; Slice1=0.7946, Slice2=0.2054
2025-07-30 09:43:00.862: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-30 09:43:02.890: Got KPIs from Qosium: [{"downlink": {"throughput": 108968.0, "latency": 10.060999870300293, "jitter": 3.239000082015991, "packetLoss": 0.0}, "uplink": {"throughput": 90052456.0, "latency": 10.956999778747559, "jitter": 0.26499998569488525, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 5.5295023918151855, "jitter": 1.680999994277954, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.74849796295166, "jitter": 2.6659998893737793, "packetLoss": 0.0}}]
2025-07-30 09:43:02.898: Got DL slice allocation; Slice1=0.7948, Slice2=0.2052
2025-07-30 09:43:02.899: Got UL slice allocation; Slice1=0.7916, Slice2=0.2084
2025-07-30 09:43:04.926: Got KPIs from Qosium: [{"downlink": {"throughput": 190216.0, "latency": 9.60200023651123, "jitter": 1.9149999618530273, "packetLoss": 0.0}, "uplink": {"throughput": 106475960.0, "latency": 13.217000007629395, "jitter": 0.8040000200271606, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.567999839782715, "jitter": 0.9120000004768372, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 8.467000007629395, "jitter": 0.9819999933242798, "packetLoss": 0.0}}]
2025-07-30 09:43:04.936: Got DL slice allocation; Slice1=0.7954, Slice2=0.2046
2025-07-30 09:43:04.936: Got UL slice allocation; Slice1=0.792, Slice2=0.208
2025-07-30 09:43:06.971: Got KPIs from Qosium: [{"downlink": {"throughput": 147048.0, "latency": 7.335999965667725, "jitter": 0.5099999904632568, "packetLoss": 0.0}, "uplink": {"throughput": 101935736.0, "latency": 10.61299991607666, "jitter": 0.6919999718666077, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.235907077789307, "jitter": 0.09399999678134918, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.451092720031738, "jitter": 1.9509999752044678, "packetLoss": 0.0}}]
2025-07-30 09:43:06.980: Got DL slice allocation; Slice1=0.7946, Slice2=0.2054
2025-07-30 09:43:06.980: Got UL slice allocation; Slice1=0.7909, Slice2=0.2091
2025-07-30 09:43:09.024: Got KPIs from Qosium: [{"downlink": {"throughput": 102824.0, "latency": 8.317999839782715, "jitter": 1.9910000562667847, "packetLoss": 0.0}, "uplink": {"throughput": 72584176.0, "latency": 11.583000183105469, "jitter": 0.6079999804496765, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 3.9700000286102295, "jitter": 0.10700000077486038, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.008999824523926, "jitter": 1.652999997138977, "packetLoss": 0.0}}]
2025-07-30 09:43:09.033: Got DL slice allocation; Slice1=0.7953, Slice2=0.2047
2025-07-30 09:43:09.033: Got UL slice allocation; Slice1=0.7917, Slice2=0.2083
2025-07-30 09:43:11.066: Got KPIs from Qosium: [{"downlink": {"throughput": 138504.0, "latency": 8.312000274658203, "jitter": 1.6699999570846558, "packetLoss": 0.0}, "uplink": {"throughput": 94245112.0, "latency": 10.345999717712402, "jitter": 0.25099998712539673, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.788432598114014, "jitter": 1.1649999618530273, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.491567611694336, "jitter": 3.124000072479248, "packetLoss": 0.0}}]
2025-07-30 09:43:11.080: Got DL slice allocation; Slice1=0.7945, Slice2=0.2055
2025-07-30 09:43:11.080: Got UL slice allocation; Slice1=0.791, Slice2=0.209
2025-07-30 09:43:13.116: Got KPIs from Qosium: [{"downlink": {"throughput": 139464.0, "latency": 10.199999809265137, "jitter": 1.7869999408721924, "packetLoss": 0.0}, "uplink": {"throughput": 102282616.0, "latency": 11.737000465393066, "jitter": 0.6349999904632568, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.433284759521484, "jitter": 0.656000018119812, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 9.631714820861816, "jitter": 3.311000108718872, "packetLoss": 0.0}}]
2025-07-30 09:43:13.125: Got DL slice allocation; Slice1=0.7947, Slice2=0.2053
2025-07-30 09:43:13.125: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-30 09:43:15.165: Got KPIs from Qosium: [{"downlink": {"throughput": 127272.0, "latency": 8.187000274658203, "jitter": 3.2709999084472656, "packetLoss": 0.0}, "uplink": {"throughput": 109370904.0, "latency": 13.243000030517578, "jitter": 0.5299999713897705, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.682412624359131, "jitter": 0.8500000238418579, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.556587219238281, "jitter": 1.0950000286102295, "packetLoss": 0.0}}]
2025-07-30 09:43:15.172: Got DL slice allocation; Slice1=0.7951, Slice2=0.2049
2025-07-30 09:43:15.173: Got UL slice allocation; Slice1=0.7914, Slice2=0.2086
2025-07-30 09:43:17.212: Got KPIs from Qosium: [{"downlink": {"throughput": 141224.0, "latency": 8.821999549865723, "jitter": 1.6339999437332153, "packetLoss": 0.0}, "uplink": {"throughput": 97474224.0, "latency": 10.925999641418457, "jitter": 0.4650000035762787, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.255000114440918, "jitter": 0.335999995470047, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 8.798999786376953, "jitter": 0.4449999928474426, "packetLoss": 0.0}}]
2025-07-30 09:43:17.221: Got DL slice allocation; Slice1=0.7954, Slice2=0.2046
2025-07-30 09:43:17.221: Got UL slice allocation; Slice1=0.792, Slice2=0.208
2025-07-30 09:43:19.261: Got KPIs from Qosium: [{"downlink": {"throughput": 140648.0, "latency": 7.355999946594238, "jitter": 2.200000047683716, "packetLoss": 0.0}, "uplink": {"throughput": 85205880.0, "latency": 12.57699966430664, "jitter": 0.4909999966621399, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.978000164031982, "jitter": 1.190000057220459, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 8.904999732971191, "jitter": 0.7440000176429749, "packetLoss": 0.0}}]
2025-07-30 09:43:19.270: Got DL slice allocation; Slice1=0.7951, Slice2=0.2049
2025-07-30 09:43:19.270: Got UL slice allocation; Slice1=0.7915, Slice2=0.2085
2025-07-30 09:43:21.309: Got KPIs from Qosium: [{"downlink": {"throughput": 110856.0, "latency": 7.784232139587402, "jitter": 1.8309999704360962, "packetLoss": 0.0}, "uplink": {"throughput": 88108328.0, "latency": 11.208767890930176, "jitter": 0.2750000059604645, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 3.866657257080078, "jitter": 0.6859999895095825, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 9.568343162536621, "jitter": 2.822000026702881, "packetLoss": 0.0}}]
2025-07-30 09:43:21.318: Got DL slice allocation; Slice1=0.7946, Slice2=0.2054
2025-07-30 09:43:21.318: Got UL slice allocation; Slice1=0.7909, Slice2=0.2091
2025-07-30 09:43:23.359: Got KPIs from Qosium: [{"downlink": {"throughput": 134184.0, "latency": 8.682000160217285, "jitter": 1.5490000247955322, "packetLoss": 0.0}, "uplink": {"throughput": 106346280.0, "latency": 11.173999786376953, "jitter": 0.7509999871253967, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.442342281341553, "jitter": 0.6700000166893005, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.84965705871582, "jitter": 1.7549999952316284, "packetLoss": 0.0}}]
2025-07-30 09:43:23.367: Got DL slice allocation; Slice1=0.7953, Slice2=0.2047
2025-07-30 09:43:23.367: Got UL slice allocation; Slice1=0.792, Slice2=0.208
2025-07-30 09:43:25.404: Got KPIs from Qosium: [{"downlink": {"throughput": 106280.0, "latency": 8.512999534606934, "jitter": 4.044000148773193, "packetLoss": 0.0}, "uplink": {"throughput": 82263584.0, "latency": 10.86400032043457, "jitter": 0.2290000021457672, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.540501594543457, "jitter": 0.14300000667572021, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.165497779846191, "jitter": 1.652999997138977, "packetLoss": 0.0}}]
2025-07-30 09:43:25.412: Got DL slice allocation; Slice1=0.7951, Slice2=0.2049
2025-07-30 09:43:25.412: Got UL slice allocation; Slice1=0.7915, Slice2=0.2085
2025-07-30 09:43:27.451: Got KPIs from Qosium: [{"downlink": {"throughput": 123112.0, "latency": 9.085000038146973, "jitter": 2.313999891281128, "packetLoss": 0.0}, "uplink": {"throughput": 107216704.0, "latency": 13.156999588012695, "jitter": 0.2540000081062317, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.577159881591797, "jitter": 1.1419999599456787, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.652839660644531, "jitter": 2.2100000381469727, "packetLoss": 0.0}}]
2025-07-30 09:43:27.459: Got DL slice allocation; Slice1=0.7947, Slice2=0.2053
2025-07-30 09:43:27.459: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-30 09:43:29.501: Got KPIs from Qosium: [{"downlink": {"throughput": 99624.0, "latency": 5.836999893188477, "jitter": 2.2699999809265137, "packetLoss": 0.0}, "uplink": {"throughput": 77328832.0, "latency": 11.677000045776367, "jitter": 0.5230000019073486, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.25160026550293, "jitter": 0.08799999952316284, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 9.575400352478027, "jitter": 2.9019999504089355, "packetLoss": 0.0}}]
2025-07-30 09:43:29.509: Got DL slice allocation; Slice1=0.795, Slice2=0.205
2025-07-30 09:43:29.509: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-30 09:43:31.546: Got KPIs from Qosium: [{"downlink": {"throughput": 118120.0, "latency": 6.986999988555908, "jitter": 3.6470000743865967, "packetLoss": 0.0}, "uplink": {"throughput": 99787016.0, "latency": 12.826000213623047, "jitter": 0.24500000476837158, "packetLoss": 0.0}}, {"downlink": {"throughput": 4656.0, "latency": 5.3470001220703125, "jitter": 1.9119999408721924, "packetLoss": 0.0}, "uplink": {"throughput": 8784.0, "latency": 10.144000053405762, "jitter": 1.3619999885559082, "packetLoss": 0.0}}]
2025-07-30 09:43:31.554: Got DL slice allocation; Slice1=0.7948, Slice2=0.2052
2025-07-30 09:43:31.554: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-30 09:43:33.589: Got KPIs from Qosium: [{"downlink": {"throughput": 176232.0, "latency": 6.51800012588501, "jitter": 1.7050000429153442, "packetLoss": 0.0}, "uplink": {"throughput": 90273720.0, "latency": 11.781000137329102, "jitter": 0.7739999890327454, "packetLoss": 0.0}}, {"downlink": {"throughput": 3720.0, "latency": 5.369999885559082, "jitter": 2.371999979019165, "packetLoss": 0.0}, "uplink": {"throughput": 8184.0, "latency": 10.04699993133545, "jitter": 2.6549999713897705, "packetLoss": 0.0}}]
2025-07-30 09:43:33.598: Got DL slice allocation; Slice1=0.7945, Slice2=0.2055
2025-07-30 09:43:33.598: Got UL slice allocation; Slice1=0.791, Slice2=0.209
2025-07-30 09:43:35.638: Got KPIs from Qosium: [{"downlink": {"throughput": 126856.0, "latency": 8.138999938964844, "jitter": 1.3630000352859497, "packetLoss": 0.0}, "uplink": {"throughput": 109150696.0, "latency": 13.343999862670898, "jitter": 0.8330000042915344, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.894244194030762, "jitter": 1.2940000295639038, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.017755508422852, "jitter": 3.7039999961853027, "packetLoss": 0.0}}]
2025-07-30 09:43:35.647: Got DL slice allocation; Slice1=0.7946, Slice2=0.2054
2025-07-30 09:43:35.647: Got UL slice allocation; Slice1=0.7907, Slice2=0.2093
2025-07-30 09:43:37.687: Got KPIs from Qosium: [{"downlink": {"throughput": 129576.0, "latency": 7.7170000076293945, "jitter": 0.5509999990463257, "packetLoss": 0.0}, "uplink": {"throughput": 108952560.0, "latency": 11.076000213623047, "jitter": 0.5249999761581421, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 3.8447184562683105, "jitter": 3.7090001106262207, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.675281524658203, "jitter": 1.715999960899353, "packetLoss": 0.0}}]
2025-07-30 09:43:37.695: Got DL slice allocation; Slice1=0.7935, Slice2=0.2065
2025-07-30 09:43:37.695: Got UL slice allocation; Slice1=0.7899, Slice2=0.2101
2025-07-30 09:43:39.729: Got KPIs from Qosium: [{"downlink": {"throughput": 116872.0, "latency": 7.569059371948242, "jitter": 2.4210000038146973, "packetLoss": 0.0}, "uplink": {"throughput": 96873240.0, "latency": 12.452940940856934, "jitter": 0.8769999742507935, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.565000057220459, "jitter": 3.378999948501587, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.843000411987305, "jitter": 3.7109999656677246, "packetLoss": 0.0}}]
2025-07-30 09:43:39.737: Got DL slice allocation; Slice1=0.7942, Slice2=0.2058
2025-07-30 09:43:39.737: Got UL slice allocation; Slice1=0.7907, Slice2=0.2093
2025-07-30 09:43:41.771: Got KPIs from Qosium: [{"downlink": {"throughput": 134696.0, "latency": 8.40268611907959, "jitter": 4.020999908447266, "packetLoss": 0.0}, "uplink": {"throughput": 111926264.0, "latency": 10.787313461303711, "jitter": 0.7009999752044678, "packetLoss": 0.0}}, {"downlink": {"throughput": 4048.0, "latency": 4.558000087738037, "jitter": 0.640999972820282, "packetLoss": 0.0}, "uplink": {"throughput": 9656.0, "latency": 10.197999954223633, "jitter": 1.5119999647140503, "packetLoss": 0.0}}]
2025-07-30 09:43:41.778: Got DL slice allocation; Slice1=0.7952, Slice2=0.2048
2025-07-30 09:43:41.778: Got UL slice allocation; Slice1=0.7917, Slice2=0.2083
2025-07-30 09:43:43.817: Got KPIs from Qosium: [{"downlink": {"throughput": 105640.0, "latency": 7.4083781242370605, "jitter": 2.759000062942505, "packetLoss": 0.0}, "uplink": {"throughput": 83862688.0, "latency": 10.124621391296387, "jitter": 0.8190000057220459, "packetLoss": 0.0}}, {"downlink": {"throughput": 3720.0, "latency": 5.179999828338623, "jitter": 0.4020000100135803, "packetLoss": 0.0}, "uplink": {"throughput": 10120.0, "latency": 11.013999938964844, "jitter": 1.309000015258789, "packetLoss": 0.0}}]
2025-07-30 09:43:43.827: Got DL slice allocation; Slice1=0.7955, Slice2=0.2045
2025-07-30 09:43:43.827: Got UL slice allocation; Slice1=0.792, Slice2=0.208
2025-07-30 09:43:45.861: Got KPIs from Qosium: [{"downlink": {"throughput": 122472.0, "latency": 7.852690696716309, "jitter": 3.26200008392334, "packetLoss": 0.0}, "uplink": {"throughput": 93948152.0, "latency": 10.068309783935547, "jitter": 0.7559999823570251, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.748028755187988, "jitter": 0.49900001287460327, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 9.627971649169922, "jitter": 2.994999885559082, "packetLoss": 0.0}}]
2025-07-30 09:43:45.870: Got DL slice allocation; Slice1=0.7954, Slice2=0.2046
2025-07-30 09:43:45.870: Got UL slice allocation; Slice1=0.792, Slice2=0.208
2025-07-30 09:43:47.907: Got KPIs from Qosium: [{"downlink": {"throughput": 105864.0, "latency": 6.917034149169922, "jitter": 3.2669999599456787, "packetLoss": 0.0}, "uplink": {"throughput": 85730888.0, "latency": 10.79196548461914, "jitter": 0.49799999594688416, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.539356708526611, "jitter": 0.7379999756813049, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.450643539428711, "jitter": 1.8660000562667847, "packetLoss": 0.0}}]
2025-07-30 09:43:47.916: Got DL slice allocation; Slice1=0.7952, Slice2=0.2048
2025-07-30 09:43:47.916: Got UL slice allocation; Slice1=0.7916, Slice2=0.2084
2025-07-30 09:43:49.947: Got KPIs from Qosium: [{"downlink": {"throughput": 126440.0, "latency": 6.953999996185303, "jitter": 2.4240000247955322, "packetLoss": 0.0}, "uplink": {"throughput": 110994904.0, "latency": 12.880000114440918, "jitter": 0.3240000009536743, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.767000198364258, "jitter": 0.3319999873638153, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.105999946594238, "jitter": 0.9700000286102295, "packetLoss": 0.0}}]
2025-07-30 09:43:49.954: Got DL slice allocation; Slice1=0.7952, Slice2=0.2048
2025-07-30 09:43:49.954: Got UL slice allocation; Slice1=0.7913, Slice2=0.2087
2025-07-30 09:43:51.992: Got KPIs from Qosium: [{"downlink": {"throughput": 139560.0, "latency": 7.3473896980285645, "jitter": 0.9229999780654907, "packetLoss": 0.0}, "uplink": {"throughput": 105597328.0, "latency": 14.57761001586914, "jitter": 0.7239999771118164, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.635381698608398, "jitter": 0.8190000057220459, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.201618194580078, "jitter": 2.561000108718872, "packetLoss": 0.0}}]
2025-07-30 09:43:52.001: Got DL slice allocation; Slice1=0.7946, Slice2=0.2054
2025-07-30 09:43:52.001: Got UL slice allocation; Slice1=0.7906, Slice2=0.2094
2025-07-30 09:43:54.044: Got KPIs from Qosium: [{"downlink": {"throughput": 124328.0, "latency": 8.333000183105469, "jitter": 1.0299999713897705, "packetLoss": 0.0}, "uplink": {"throughput": 98154064.0, "latency": 13.640999794006348, "jitter": 0.5189999938011169, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 5.070000171661377, "jitter": 0.8009999990463257, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.244000434875488, "jitter": 3.1600000858306885, "packetLoss": 0.0}}]
2025-07-30 09:43:54.053: Got DL slice allocation; Slice1=0.7944, Slice2=0.2056
2025-07-30 09:43:54.053: Got UL slice allocation; Slice1=0.7904, Slice2=0.2096
2025-07-30 09:43:56.099: Got KPIs from Qosium: [{"downlink": {"throughput": 115016.0, "latency": 7.242496967315674, "jitter": 1.75, "packetLoss": 0.0}, "uplink": {"throughput": 91726712.0, "latency": 12.111503601074219, "jitter": 0.28999999165534973, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.659999847412109, "jitter": 0.6679999828338623, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.211000442504883, "jitter": 2.4660000801086426, "packetLoss": 0.0}}]
2025-07-30 09:43:56.108: Got DL slice allocation; Slice1=0.7948, Slice2=0.2052
2025-07-30 09:43:56.108: Got UL slice allocation; Slice1=0.791, Slice2=0.209
2025-07-30 09:43:58.144: Got KPIs from Qosium: [{"downlink": {"throughput": 121768.0, "latency": 10.149999618530273, "jitter": 0.24400000274181366, "packetLoss": 0.0}, "uplink": {"throughput": 90917048.0, "latency": 13.253999710083008, "jitter": 0.7919999957084656, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.5157790184021, "jitter": 0.024000000208616257, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 11.11522102355957, "jitter": 4.0980000495910645, "packetLoss": 0.0}}]
2025-07-30 09:43:58.153: Got DL slice allocation; Slice1=0.7937, Slice2=0.2063
2025-07-30 09:43:58.153: Got UL slice allocation; Slice1=0.7897, Slice2=0.2103
2025-07-30 09:44:00.186: Got KPIs from Qosium: [{"downlink": {"throughput": 115624.0, "latency": 6.802169322967529, "jitter": 1.8450000286102295, "packetLoss": 0.0}, "uplink": {"throughput": 84857776.0, "latency": 11.880830764770508, "jitter": 0.3100000023841858, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 5.440999984741211, "jitter": 0.00800000037997961, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.343000411987305, "jitter": 2.631999969482422, "packetLoss": 0.0}}]
2025-07-30 09:44:00.194: Got DL slice allocation; Slice1=0.795, Slice2=0.205
2025-07-30 09:44:00.194: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-30 09:44:02.228: Got KPIs from Qosium: [{"downlink": {"throughput": 129992.0, "latency": 6.226062774658203, "jitter": 1.2649999856948853, "packetLoss": 0.0}, "uplink": {"throughput": 108773160.0, "latency": 9.72293758392334, "jitter": 0.28600001335144043, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.349999904632568, "jitter": 0.9829999804496765, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.515999794006348, "jitter": 1.694000005722046, "packetLoss": 0.0}}]
2025-07-30 09:44:02.236: Got DL slice allocation; Slice1=0.7947, Slice2=0.2053
2025-07-30 09:44:02.236: Got UL slice allocation; Slice1=0.7912, Slice2=0.2088
2025-07-30 09:44:04.278: Got KPIs from Qosium: [{"downlink": {"throughput": 116264.0, "latency": 7.929028511047363, "jitter": 3.683000087738037, "packetLoss": 0.0}, "uplink": {"throughput": 102380176.0, "latency": 11.401971817016602, "jitter": 0.6869999766349792, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.837534427642822, "jitter": 0.11999999731779099, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.450465202331543, "jitter": 3.3910000324249268, "packetLoss": 0.0}}]
2025-07-30 09:44:04.287: Got DL slice allocation; Slice1=0.7953, Slice2=0.2047
2025-07-30 09:44:04.287: Got UL slice allocation; Slice1=0.7916, Slice2=0.2084
2025-07-30 09:44:06.331: Got KPIs from Qosium: [{"downlink": {"throughput": 179048.0, "latency": 7.578869342803955, "jitter": 0.6439999938011169, "packetLoss": 0.0}, "uplink": {"throughput": 90720520.0, "latency": 13.342130661010742, "jitter": 1.2569999694824219, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.6620001792907715, "jitter": 0.41200000047683716, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.854000091552734, "jitter": 2.2829999923706055, "packetLoss": 0.0}}]
2025-07-30 09:44:06.341: Got DL slice allocation; Slice1=0.795, Slice2=0.205
2025-07-30 09:44:06.341: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-30 09:44:08.387: Got KPIs from Qosium: [{"downlink": {"throughput": 3304.0, "latency": 5.877306938171387, "jitter": 0.5049999952316284, "packetLoss": 0.0}, "uplink": {"throughput": 5880.0, "latency": 9.489692687988281, "jitter": 1.8200000524520874, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 5.169000148773193, "jitter": 0.7509999871253967, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.239999771118164, "jitter": 2.00600004196167, "packetLoss": 0.0}}]
2025-07-30 09:44:08.397: Got DL slice allocation; Slice1=0.6075, Slice2=0.3925
2025-07-30 09:44:08.397: Got UL slice allocation; Slice1=0.6414, Slice2=0.3586
2025-07-30 09:44:10.441: Got KPIs from Qosium: [{"downlink": {"throughput": 2696.0, "latency": 5.024216175079346, "jitter": 0.23499999940395355, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 10.11678409576416, "jitter": 1.4539999961853027, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 5.175000190734863, "jitter": 0.6449999809265137, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.121999740600586, "jitter": 3.0280001163482666, "packetLoss": 0.0}}]
2025-07-30 09:44:10.450: Got DL slice allocation; Slice1=0.5299, Slice2=0.4701
2025-07-30 09:44:10.450: Got UL slice allocation; Slice1=0.5706, Slice2=0.4294
2025-07-30 09:44:12.486: Got KPIs from Qosium: [{"downlink": {"throughput": 2696.0, "latency": 5.205234050750732, "jitter": 0.28600001335144043, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 10.602766036987305, "jitter": 1.069000005722046, "packetLoss": 0.0}}, {"downlink": {"throughput": 2944.0, "latency": 4.902899742126465, "jitter": 1.2230000495910645, "packetLoss": 0.0}, "uplink": {"throughput": 4184.0, "latency": 11.261099815368652, "jitter": 2.575000047683716, "packetLoss": 0.0}}]
2025-07-30 09:44:12.494: Got DL slice allocation; Slice1=0.5242, Slice2=0.4758
2025-07-30 09:44:12.494: Got UL slice allocation; Slice1=0.5649, Slice2=0.4351
2025-07-30 09:44:14.545: Got KPIs from Qosium: [{"downlink": {"throughput": 3528.0, "latency": 5.37173318862915, "jitter": 0.9580000042915344, "packetLoss": 0.0}, "uplink": {"throughput": 5912.0, "latency": 10.62226676940918, "jitter": 2.2179999351501465, "packetLoss": 0.0}}, {"downlink": {"throughput": 2336.0, "latency": 4.421000003814697, "jitter": 0.09600000083446503, "packetLoss": 0.0}, "uplink": {"throughput": 4408.0, "latency": 9.206000328063965, "jitter": 1.128000020980835, "packetLoss": 0.0}}]
2025-07-30 09:44:14.555: Got DL slice allocation; Slice1=0.6643, Slice2=0.3357
2025-07-30 09:44:14.555: Got UL slice allocation; Slice1=0.6886, Slice2=0.3114
2025-07-30 09:44:16.587: Got KPIs from Qosium: [{"downlink": {"throughput": 3304.0, "latency": 5.748997688293457, "jitter": 0.9120000004768372, "packetLoss": 0.0}, "uplink": {"throughput": 5880.0, "latency": 10.034002304077148, "jitter": 1.6660000085830688, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.328000068664551, "jitter": 0.17800000309944153, "packetLoss": 0.0}, "uplink": {"throughput": 4184.0, "latency": 9.848999977111816, "jitter": 2.2070000171661377, "packetLoss": 0.0}}]
2025-07-30 09:44:16.595: Got DL slice allocation; Slice1=0.6478, Slice2=0.3522
2025-07-30 09:44:16.595: Got UL slice allocation; Slice1=0.6759, Slice2=0.3241
2025-07-30 09:44:18.625: Got KPIs from Qosium: [{"downlink": {"throughput": 2696.0, "latency": 5.066999912261963, "jitter": 0.5950000286102295, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 10.876999855041504, "jitter": 2.3970000743865967, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.677075386047363, "jitter": 0.9309999942779541, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.400924682617188, "jitter": 2.0339999198913574, "packetLoss": 0.0}}]
2025-07-30 09:44:18.635: Got DL slice allocation; Slice1=0.5288, Slice2=0.4712
2025-07-30 09:44:18.635: Got UL slice allocation; Slice1=0.5686, Slice2=0.4314
"""

def parse_and_prepare_data(log_text):
    """
    Parses the raw log text into a structured pandas DataFrame. This version is
    robust and avoids complex regular expressions.
    """
    records = []
    current_kpi_data = None
    current_timestamp = None

    for line in log_text.strip().split('\n'):
        try:
            timestamp_str = line[:23]
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            
            if "Got KPIs from Qosium:" in line:
                # This is a new record, store the KPIs
                json_part = line.split('Got KPIs from Qosium: ')[1]
                kpis = json.loads(json_part)
                current_kpi_data = kpis
                current_timestamp = timestamp
                
            elif "Got DL slice allocation;" in line:
                # This line contains the DL allocation for the current record
                parts = line.split(';')[-1].strip().split(',')
                dl_alloc_s1 = float(parts[0].split('=')[1])
                dl_alloc_s2 = float(parts[1].split('=')[1])
                
            elif "Got UL slice allocation;" in line:
                # This line has the UL allocation and completes the record
                parts = line.split(';')[-1].strip().split(',')
                ul_alloc_s1 = float(parts[0].split('=')[1])
                ul_alloc_s2 = float(parts[1].split('=')[1])
                
                # Assemble and save the completed record
                if current_kpi_data and current_timestamp:
                    record = {
                        'timestamp': current_timestamp,
                        'dl_thr_s1': current_kpi_data[0]['downlink']['throughput'],
                        'ul_thr_s1': current_kpi_data[0]['uplink']['throughput'],
                        'dl_thr_s2': current_kpi_data[1]['downlink']['throughput'],
                        'ul_thr_s2': current_kpi_data[1]['uplink']['throughput'],
                        'dl_alloc_s1': dl_alloc_s1,
                        'dl_alloc_s2': dl_alloc_s2,
                        'ul_alloc_s1': ul_alloc_s1,
                        'ul_alloc_s2': ul_alloc_s2,
                    }
                    records.append(record)
                
                # Reset for the next record
                current_kpi_data = None
                current_timestamp = None

        except (ValueError, IndexError, json.JSONDecodeError, KeyError) as e:
            # Silently skip any line that doesn't match the expected format
            continue
            
    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index('timestamp', inplace=True)
    return df


def plot_live_results(df):
    """
    Generates a professional, two-panel graph visualizing the AI's performance.
    """
    if df.empty:
        print("Error: DataFrame is empty. Cannot generate plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('Live Network Performance of Nuanced AI Slicing Agent', fontsize=18, weight='bold')

    # --- Top Panel: Network Throughput (The "Why") ---
    ax1.set_title('Observed Network Conditions (Throughput)', fontsize=14)
    ax1.plot(df.index, df['dl_thr_s1'], label='Slice 1 DL Throughput', color='blue', linewidth=2)
    ax1.plot(df.index, df['ul_thr_s1'], label='Slice 1 UL Throughput', color='deepskyblue', linestyle='--')
    ax1.plot(df.index, df['dl_thr_s2'], label='Slice 2 DL Throughput', color='red', linewidth=2)
    ax1.plot(df.index, df['ul_thr_s2'], label='Slice 2 UL Throughput', color='salmon', linestyle='--')
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Throughput (log scale, bps)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # --- Bottom Panel: AI Allocation Decisions (The "What") ---
    ax2.set_title("AI Agent's Allocation Decisions", fontsize=14)
    ax2.plot(df.index, df['dl_alloc_s1'], label='Slice 1 DL Allocation', color='blue', linewidth=2.5, marker='.', markersize=8)
    ax2.plot(df.index, df['ul_alloc_s1'], label='Slice 1 UL Allocation', color='deepskyblue', linestyle=':', marker='x')
    
    ax2.set_ylabel('Allocation to Slice 1', fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.axhline(0.5, color='grey', linestyle='--', label='50/50 Split')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', linewidth=0.5)

    # --- Annotations for the "Three Acts" ---
    phase1_end = datetime.strptime('2025-07-30 09:42:41', '%Y-%m-%d %H:%M:%S')
    phase2_end = datetime.strptime('2025-07-30 09:44:07', '%Y-%m-%d %H:%M:%S')
    
    for ax in [ax1, ax2]:
        ax.axvspan(df.index.min(), phase1_end, facecolor='green', alpha=0.1)
        ax.axvspan(phase1_end, phase2_end, facecolor='orange', alpha=0.1)
        ax.axvspan(phase2_end, df.index.max(), facecolor='green', alpha=0.1)

    ax2.text(datetime.strptime('2025-07-30 09:42:34', '%Y-%m-%d %H:%M:%S'), 0.05, 'Phase 1: Competitive\n(Nuanced Allocations)', ha='center', fontsize=12, style='italic', color='darkgreen')
    ax2.text(datetime.strptime('2025-07-30 09:43:20', '%Y-%m-%d %H:%M:%S'), 0.05, 'Phase 2: S1 Dominant\n(Decisive Allocations)', ha='center', fontsize=12, style='italic', color='darkgoldenrod')
    ax2.text(datetime.strptime('2025-07-30 09:44:13', '%Y-%m-%d %H:%M:%S'), 0.05, 'Phase 3: Recovery\n(Nuanced Allocations)', ha='center', fontsize=12, style='italic', color='darkgreen')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xlabel('Time', fontsize=12)
    plt.xticks(rotation=45)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_filename = "live_network_performance.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*50)
    print("✅ Graph saved successfully!")
    print(f"   Location: {os.path.abspath(save_filename)}")
    print("="*50)

    print("\nDisplaying plot window. Close the window to exit the script.")
    plt.show(block=True) # Use block=True for scripts

# --- Main execution block ---
if __name__ == "__main__":
    try:
        live_data_df = parse_and_prepare_data(LOG_DATA)
        print("Log data parsing successfully.")
        
        if not live_data_df.empty:
            print(f"Found {len(live_data_df)} complete log entries.")
            
            # --- ACTION: Add Console Output ---
            print("\n--- First 5 Parsed Data Points ---")
            print(live_data_df.head())
            print("----------------------------------\n")
            
            # --- ACTION: Add CSV Output ---
            csv_filename = "live_results.csv"
            live_data_df.to_csv(csv_filename)
            print("✅ Full dataset saved to CSV!")
            print(f"   Location: {os.path.abspath(csv_filename)}")
            print("="*50)

            plot_live_results(live_data_df)
        else:
            print("No valid log entries were found to plot.")
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please ensure pandas and matplotlib are installed (`pip install pandas matplotlib`).")
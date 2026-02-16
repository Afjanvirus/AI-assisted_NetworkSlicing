import json
import pandas as pd
from datetime import datetime

# --- 1. Raw Log Data ---
# All log entries from both sessions are included here.
LOG_DATA_1 = """
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
"""

LOG_DATA_2 = """
2025-07-31 10:05:10.945: Got KPIs from Qosium: [{"downlink": {"throughput": 3304.0, "latency": 6.422043, "jitter": 1.503, "packetLoss": 0.0}, "uplink": {"throughput": 5880.0, "latency": 9.491957, "jitter": 1.798, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.492, "jitter": 0.0, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 9.971, "jitter": 1.949, "packetLoss": 0.0}}]
2025-07-31 10:05:10.964: Got DL slice allocation; Slice1=0.6989, Slice2=0.3011
2025-07-31 10:05:10.964: Got UL slice allocation; Slice1=0.7181, Slice2=0.2819
2025-07-31 10:05:12.996: Got KPIs from Qosium: [{"downlink": {"throughput": 2696.0, "latency": 5.432, "jitter": 1.207, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 9.326, "jitter": 1.656, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.6343484, "jitter": 0.024, "packetLoss": 0.0}, "uplink": {"throughput": 4032.0, "latency": 10.380651, "jitter": 1.3, "packetLoss": 0.0}}]
2025-07-31 10:05:13.004: Got DL slice allocation; Slice1=0.6359, Slice2=0.3641
2025-07-31 10:05:13.004: Got UL slice allocation; Slice1=0.6655, Slice2=0.3345
2025-07-31 10:05:15.044: Got KPIs from Qosium: [{"downlink": {"throughput": 3912.0, "latency": 5.6795874, "jitter": 0.853, "packetLoss": 0.0}, "uplink": {"throughput": 6904.0, "latency": 9.263412, "jitter": 1.34, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 5.275622, "jitter": 1.585, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.522378, "jitter": 1.621, "packetLoss": 0.0}}]
2025-07-31 10:05:15.052: Got DL slice allocation; Slice1=0.6332, Slice2=0.3668
2025-07-31 10:05:15.052: Got UL slice allocation; Slice1=0.6647, Slice2=0.3353
2025-07-31 10:05:17.083: Got KPIs from Qosium: [{"downlink": {"throughput": 2696.0, "latency": 5.4536366, "jitter": 1.512, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 9.522364, "jitter": 0.974, "packetLoss": 0.0}}, {"downlink": {"throughput": 3552.0, "latency": 5.714, "jitter": 0.926, "packetLoss": 0.0}, "uplink": {"throughput": 6488.0, "latency": 11.153, "jitter": 1.14, "packetLoss": 0.0}}]
2025-07-31 10:05:17.090: Got DL slice allocation; Slice1=0.4084, Slice2=0.5916
2025-07-31 10:05:17.090: Got UL slice allocation; Slice1=0.4545, Slice2=0.5455
2025-07-31 10:05:19.125: Got KPIs from Qosium: [{"downlink": {"throughput": 4488.0, "latency": 6.824, "jitter": 1.755, "packetLoss": 0.0}, "uplink": {"throughput": 7256.0, "latency": 9.802, "jitter": 1.662, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.476, "jitter": 0.059, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 9.94, "jitter": 1.511, "packetLoss": 0.0}}]
2025-07-31 10:05:19.134: Got DL slice allocation; Slice1=0.7481, Slice2=0.2519
2025-07-31 10:05:19.134: Got UL slice allocation; Slice1=0.7545, Slice2=0.2455
2025-07-31 10:05:21.165: Got KPIs from Qosium: [{"downlink": {"throughput": 2696.0, "latency": 5.431000232696533, "jitter": 0.5400000214576721, "packetLoss": 0.0}, "uplink": {"throughput": 4856.0, "latency": 9.008000373840332, "jitter": 2.115999937057495, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.593990325927734, "jitter": 0.01600000075995922, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.178009033203125, "jitter": 1.5549999475479126, "packetLoss": 0.0}}]
2025-07-31 10:05:21.174: Got DL slice allocation; Slice1=0.6406, Slice2=0.3594
2025-07-31 10:05:21.174: Got UL slice allocation; Slice1=0.6693, Slice2=0.3307
2025-07-31 10:05:23.222: Got KPIs from Qosium: [{"downlink": {"throughput": 3304.0, "latency": 5.930020332336426, "jitter": 0.5400000214576721, "packetLoss": 0.0}, "uplink": {"throughput": 5880.0, "latency": 9.164979934692383, "jitter": 1.1260000467300415, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.600925445556641, "jitter": 0.8980000019073486, "packetLoss": 0.0}, "uplink": {"throughput": 3816.0, "latency": 10.56607437133789, "jitter": 1.5240000486373901, "packetLoss": 0.0}}]
2025-07-31 10:05:23.231: Got DL slice allocation; Slice1=0.6826, Slice2=0.3174
2025-07-31 10:05:23.231: Got UL slice allocation; Slice1=0.7053, Slice2=0.2947
2025-07-31 10:05:25.266: Got KPIs from Qosium: [{"downlink": {"throughput": 5480.0, "latency": 5.889590263366699, "jitter": 1.0770000219345093, "packetLoss": 0.0}, "uplink": {"throughput": 7816.0, "latency": 9.300409317016602, "jitter": 1.1660000085830688, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.732916831970215, "jitter": 1.0260000228881836, "packetLoss": 0.0}, "uplink": {"throughput": 4120.0, "latency": 9.640083312988281, "jitter": 1.7300000190734863, "packetLoss": 0.0}}]
2025-07-31 10:05:25.274: Got DL slice allocation; Slice1=0.7341, Slice2=0.2659
2025-07-31 10:05:25.274: Got UL slice allocation; Slice1=0.7449, Slice2=0.2551
2025-07-31 10:05:27.307: Got KPIs from Qosium: [{"downlink": {"throughput": 3304.0, "latency": 5.508138179779053, "jitter": 0.35100001096725464, "packetLoss": 0.0}, "uplink": {"throughput": 5880.0, "latency": 9.830862045288086, "jitter": 1.5379999876022339, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 5.195924758911133, "jitter": 0.6759999990463257, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.25307559967041, "jitter": 1.187000036239624, "packetLoss": 0.0}}]
2025-07-31 10:05:27.315: Got DL slice allocation; Slice1=0.6859, Slice2=0.3141
2025-07-31 10:05:27.315: Got UL slice allocation; Slice1=0.7077, Slice2=0.2923
2025-07-31 10:05:29.351: Got KPIs from Qosium: [{"downlink": {"throughput": 5064.0, "latency": 5.828000068664551, "jitter": 0.6710000038146973, "packetLoss": 0.0}, "uplink": {"throughput": 6280.0, "latency": 9.770999908447266, "jitter": 1.6820000410079956, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.088011264801025, "jitter": 0.5419999957084656, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.722989082336426, "jitter": 1.2970000505447388, "packetLoss": 0.0}}]
2025-07-31 10:05:29.359: Got DL slice allocation; Slice1=0.7371, Slice2=0.2629
2025-07-31 10:05:29.359: Got UL slice allocation; Slice1=0.7459, Slice2=0.2541
2025-07-31 10:05:31.398: Got KPIs from Qosium: [{"downlink": {"throughput": 6664.0, "latency": 6.918000221252441, "jitter": 1.4079999923706055, "packetLoss": 0.0}, "uplink": {"throughput": 9432.0, "latency": 9.753999710083008, "jitter": 1.9889999628067017, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 3.9531891345977783, "jitter": 2.4579999446868896, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.201810836791992, "jitter": 1.3849999904632568, "packetLoss": 0.0}}]
2025-07-31 10:05:31.406: Got DL slice allocation; Slice1=0.7622, Slice2=0.2378
2025-07-31 10:05:31.406: Got UL slice allocation; Slice1=0.7653, Slice2=0.2347
2025-07-31 10:05:33.448: Got KPIs from Qosium: [{"downlink": {"throughput": 15048.0, "latency": 6.478000164031982, "jitter": 1.277999997138977, "packetLoss": 0.0}, "uplink": {"throughput": 21976.0, "latency": 9.770000457763672, "jitter": 1.5019999742507935, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.805112361907959, "jitter": 1.3940000534057617, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.197887420654297, "jitter": 1.8389999866485596, "packetLoss": 0.0}}]
2025-07-31 10:05:33.458: Got DL slice allocation; Slice1=0.7901, Slice2=0.2099
2025-07-31 10:05:33.458: Got UL slice allocation; Slice1=0.7866, Slice2=0.2134
2025-07-31 10:05:35.505: Got KPIs from Qosium: [{"downlink": {"throughput": 175392.0, "latency": 6.985000133514404, "jitter": 2.1029999256134033, "packetLoss": 0.0}, "uplink": {"throughput": 76280656.0, "latency": 15.298999786376953, "jitter": 0.7549999952316284, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.186156272888184, "jitter": 0.017999999225139618, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.716843605041504, "jitter": 1.3630000352859497, "packetLoss": 0.0}}]
2025-07-31 10:05:35.505: Got DL slice allocation; Slice1=0.7954, Slice2=0.2046
2025-07-31 10:05:35.505: Got UL slice allocation; Slice1=0.7915, Slice2=0.2085
2025-07-31 10:05:37.542: Got KPIs from Qosium: [{"downlink": {"throughput": 128272.0, "latency": 8.54800033569336, "jitter": 2.614000082015991, "packetLoss": 0.0}, "uplink": {"throughput": 99635672.0, "latency": 13.053999900817871, "jitter": 0.21899999678134918, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 3.882999897003174, "jitter": 0.6489999890327454, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 9.270999908447266, "jitter": 1.687000036239624, "packetLoss": 0.0}}]
2025-07-31 10:05:37.552: Got DL slice allocation; Slice1=0.7949, Slice2=0.2051
2025-07-31 10:05:37.553: Got UL slice allocation; Slice1=0.7911, Slice2=0.2089
2025-07-31 10:05:39.590: Got KPIs from Qosium: [{"downlink": {"throughput": 135400.0, "latency": 7.756999969482422, "jitter": 2.380000114440918, "packetLoss": 0.0}, "uplink": {"throughput": 113814776.0, "latency": 12.843999862670898, "jitter": 0.42100000381469727, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 3.8656980991363525, "jitter": 0.9070000052452087, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.368302345275879, "jitter": 1.253999948501587, "packetLoss": 0.0}}]
2025-07-31 10:05:39.599: Got DL slice allocation; Slice1=0.795, Slice2=0.205
2025-07-31 10:05:39.599: Got UL slice allocation; Slice1=0.7914, Slice2=0.2086
2025-07-31 10:05:41.638: Got KPIs from Qosium: [{"downlink": {"throughput": 111520.0, "latency": 8.826000213623047, "jitter": 2.2300000190734863, "packetLoss": 0.0}, "uplink": {"throughput": 82994432.0, "latency": 12.927000045776367, "jitter": 0.257999986410141, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.297999858856201, "jitter": 0.8859999775886536, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 9.607999801635742, "jitter": 1.4639999866485596, "packetLoss": 0.0}}]
2025-07-31 10:05:41.646: Got DL slice allocation; Slice1=0.7949, Slice2=0.2051
2025-07-31 10:05:41.647: Got UL slice allocation; Slice1=0.7912, Slice2=0.2088
2025-07-31 10:05:43.688: Got KPIs from Qosium: [{"downlink": {"throughput": 123528.0, "latency": 5.021999835968018, "jitter": 1.9620000123977661, "packetLoss": 0.0}, "uplink": {"throughput": 110859512.0, "latency": 13.175999641418457, "jitter": 1.0839999914169312, "packetLoss": 0.0}}, {"downlink": {"throughput": 1920.0, "latency": 4.140115737915039, "jitter": 0.257999986410141, "packetLoss": 0.0}, "uplink": {"throughput": 3880.0, "latency": 10.717884063720703, "jitter": 1.2120000123977661, "packetLoss": 0.0}}]
2025-07-31 10:05:43.698: Got DL slice allocation; Slice1=0.7956, Slice2=0.2044
2025-07-31 10:05:43.698: Got UL slice allocation; Slice1=0.7919, Slice2=0.2081
2025-07-31 10:05:45.738: Got KPIs from Qosium: [{"downlink": {"throughput": 103008.0, "latency": 5.2854814529418945, "jitter": 0.1979999989271164, "packetLoss": 0.0}, "uplink": {"throughput": 78359232.0, "latency": 13.530518531799316, "jitter": 0.2939999997615814, "packetLoss": 0.0}}, {"downlink": {"throughput": 4048.0, "latency": 4.5567169189453125, "jitter": 0.22200000286102295, "packetLoss": 0.0}, "uplink": {"throughput": 9656.0, "latency": 10.27828311920166, "jitter": 0.9950000047683716, "packetLoss": 0.0}}]
2025-07-31 10:05:45.747: Got DL slice allocation; Slice1=0.7942, Slice2=0.2058
2025-07-31 10:05:45.747: Got UL slice allocation; Slice1=0.79, Slice2=0.21
2025-07-31 10:05:47.792: Got KPIs from Qosium: [{"downlink": {"throughput": 121224.0, "latency": 5.888999938964844, "jitter": 2.3980000019073486, "packetLoss": 0.0}, "uplink": {"throughput": 98214928.0, "latency": 13.984999656677246, "jitter": 0.28999999165534973, "packetLoss": 0.0}}, {"downlink": {"throughput": 38536.0, "latency": 5.085000038146973, "jitter": 0.7210000157356262, "packetLoss": 0.0}, "uplink": {"throughput": 64088.0, "latency": 10.267999649047852, "jitter": 1.5019999742507935, "packetLoss": 0.0}}]
2025-07-31 10:05:47.801: Got DL slice allocation; Slice1=0.7926, Slice2=0.2074
2025-07-31 10:05:47.801: Got UL slice allocation; Slice1=0.7879, Slice2=0.2121
2025-07-31 10:05:49.834: Got KPIs from Qosium: [{"downlink": {"throughput": 113864.0, "latency": 7.090000152587891, "jitter": 3.003000020980835, "packetLoss": 0.0}, "uplink": {"throughput": 92064600.0, "latency": 14.489999771118164, "jitter": 0.2770000100135803, "packetLoss": 0.0}}, {"downlink": {"throughput": 2528.0, "latency": 4.515856742858887, "jitter": 0.7110000252723694, "packetLoss": 0.0}, "uplink": {"throughput": 4904.0, "latency": 10.564143180847168, "jitter": 1.99399995803833, "packetLoss": 0.0}}]
2025-07-31 10:05:49.843: Got DL slice allocation; Slice1=0.7949, Slice2=0.2051
2025-07-31 10:05:49.843: Got UL slice allocation; Slice1=0.791, Slice2=0.209
2025-07-31 10:05:51.877: Got KPIs from Qosium: [{"downlink": {"throughput": 131848.0, "latency": 7.604000091552734, "jitter": 4.230999946594238, "packetLoss": 0.0}, "uplink": {"throughput": 107041080.0, "latency": 13.428000450134277, "jitter": 0.6470000147819519, "packetLoss": 0.0}}, {"downlink": {"throughput": 202848.0, "latency": 6.330999851226807, "jitter": 2.4609999656677246, "packetLoss": 0.0}, "uplink": {"throughput": 25137480.0, "latency": 12.937000274658203, "jitter": 0.8429999947547913, "packetLoss": 0.0}}]
2025-07-31 10:05:51.884: Got DL slice allocation; Slice1=0.6209, Slice2=0.3791
2025-07-31 10:05:51.885: Got UL slice allocation; Slice1=0.652, Slice2=0.348
2025-07-31 10:05:53.920: Got KPIs from Qosium: [{"downlink": {"throughput": 143272.0, "latency": 9.350000381469727, "jitter": 0.26600000262260437, "packetLoss": 0.0}, "uplink": {"throughput": 87996504.0, "latency": 14.909000396728516, "jitter": 0.6980000138282776, "packetLoss": 0.0}}, {"downlink": {"throughput": 171840.0, "latency": 7.576000213623047, "jitter": 0.6349999904632568, "packetLoss": 0.0}, "uplink": {"throughput": 43797208.0, "latency": 13.786999702453613, "jitter": 0.49300000071525574, "packetLoss": 0.0}}]
2025-07-31 10:05:53.929: Got DL slice allocation; Slice1=0.5927, Slice2=0.4073
2025-07-31 10:05:53.929: Got UL slice allocation; Slice1=0.6268, Slice2=0.3732
2025-07-31 10:05:55.973: Got KPIs from Qosium: [{"downlink": {"throughput": 139784.0, "latency": 6.573999881744385, "jitter": 2.180999994277954, "packetLoss": 0.0}, "uplink": {"throughput": 86391096.0, "latency": 14.666999816894531, "jitter": 0.3140000104904175, "packetLoss": 0.0}}, {"downlink": {"throughput": 215424.0, "latency": 7.150000095367432, "jitter": 4.818999767303467, "packetLoss": 0.0}, "uplink": {"throughput": 37742328.0, "latency": 13.89900016784668, "jitter": 0.5669999718666077, "packetLoss": 0.0}}]
2025-07-31 10:05:55.982: Got DL slice allocation; Slice1=0.543, Slice2=0.457
2025-07-31 10:05:55.982: Got UL slice allocation; Slice1=0.5801, Slice2=0.4199
2025-07-31 10:05:58.026: Got KPIs from Qosium: [{"downlink": {"throughput": 118536.0, "latency": 8.557000160217285, "jitter": 2.617000102996826, "packetLoss": 0.0}, "uplink": {"throughput": 79998232.0, "latency": 15.916999816894531, "jitter": 0.3199999928474426, "packetLoss": 0.0}}, {"downlink": {"throughput": 298080.0, "latency": 6.229000091552734, "jitter": 2.5260000228881836, "packetLoss": 0.0}, "uplink": {"throughput": 53062640.0, "latency": 14.977999687194824, "jitter": 0.3889999985694885, "packetLoss": 0.0}}]
2025-07-31 10:05:58.036: Got DL slice allocation; Slice1=0.4367, Slice2=0.5633
2025-07-31 10:05:58.036: Got UL slice allocation; Slice1=0.4801, Slice2=0.5199
2025-07-31 10:06:00.074: Got KPIs from Qosium: [{"downlink": {"throughput": 119528.0, "latency": 9.244999885559082, "jitter": 3.441999912261963, "packetLoss": 0.0}, "uplink": {"throughput": 56144256.0, "latency": 15.302000045776367, "jitter": 0.5339999794960022, "packetLoss": 0.0}}, {"downlink": {"throughput": 292672.0, "latency": 8.11400032043457, "jitter": 3.00600004196167, "packetLoss": 0.0}, "uplink": {"throughput": 69241224.0, "latency": 15.852999687194824, "jitter": 0.32199999690055847, "packetLoss": 0.0}}]
2025-07-31 10:06:00.083: Got DL slice allocation; Slice1=0.322, Slice2=0.678
2025-07-31 10:06:00.083: Got UL slice allocation; Slice1=0.3692, Slice2=0.6308
2025-07-31 10:06:02.125: Got KPIs from Qosium: [{"downlink": {"throughput": 112552.0, "latency": 6.820000171661377, "jitter": 2.5759999752044678, "packetLoss": 0.0}, "uplink": {"throughput": 44048488.0, "latency": 13.854000091552734, "jitter": 0.5540000200271606, "packetLoss": 0.0}}, {"downlink": {"throughput": 253376.0, "latency": 5.359000205993652, "jitter": 2.1389999389648438, "packetLoss": 0.0}, "uplink": {"throughput": 67817120.0, "latency": 15.258999824523926, "jitter": 0.3269999921321869, "packetLoss": 0.0}}]
2025-07-31 10:06:02.134: Got DL slice allocation; Slice1=0.2855, Slice2=0.7145
2025-07-31 10:06:02.134: Got UL slice allocation; Slice1=0.3327, Slice2=0.6673
2025-07-31 10:06:04.176: Got KPIs from Qosium: [{"downlink": {"throughput": 104040.0, "latency": 8.468000411987305, "jitter": 3.3540000915527344, "packetLoss": 0.0}, "uplink": {"throughput": 44109024.0, "latency": 13.543000221252441, "jitter": 0.4359999895095825, "packetLoss": 0.0}}, {"downlink": {"throughput": 364448.0, "latency": 4.388999938964844, "jitter": 3.2839999198913574, "packetLoss": 0.0}, "uplink": {"throughput": 80468448.0, "latency": 15.984000205993652, "jitter": 0.27900001406669617, "packetLoss": 0.0}}]
2025-07-31 10:06:04.185: Got DL slice allocation; Slice1=0.2329, Slice2=0.7671
2025-07-31 10:06:04.185: Got UL slice allocation; Slice1=0.2753, Slice2=0.7247
2025-07-31 10:06:06.225: Got KPIs from Qosium: [{"downlink": {"throughput": 96264.0, "latency": 7.122000217437744, "jitter": 1.6019999980926514, "packetLoss": 0.0}, "uplink": {"throughput": 33257032.0, "latency": 13.800999641418457, "jitter": 0.6869999766349792, "packetLoss": 0.0}}, {"downlink": {"throughput": 394208.0, "latency": 7.293000221252441, "jitter": 3.868000030517578, "packetLoss": 0.0}, "uplink": {"throughput": 96240944.0, "latency": 16.79400062561035, "jitter": 0.24699999392032623, "packetLoss": 0.0}}]
2025-07-31 10:06:06.234: Got DL slice allocation; Slice1=0.2083, Slice2=0.7917
2025-07-31 10:06:06.234: Got UL slice allocation; Slice1=0.2353, Slice2=0.7647
2025-07-31 10:06:08.275: Got KPIs from Qosium: [{"downlink": {"throughput": 95496.0, "latency": 7.25, "jitter": 3.49399995803833, "packetLoss": 0.0}, "uplink": {"throughput": 29017944.0, "latency": 13.463000297546387, "jitter": 0.6380000114440918, "packetLoss": 0.0}}, {"downlink": {"throughput": 397120.0, "latency": 6.916999816894531, "jitter": 2.5369999408721924, "packetLoss": 0.0}, "uplink": {"throughput": 100925064.0, "latency": 19.638999938964844, "jitter": 0.7120000123977661, "packetLoss": 0.0}}]
2025-07-31 10:06:08.284: Got DL slice allocation; Slice1=0.209, Slice2=0.791
2025-07-31 10:06:08.285: Got UL slice allocation; Slice1=0.2372, Slice2=0.7628
2025-07-31 10:06:10.322: Got KPIs from Qosium: [{"downlink": {"throughput": 79016.0, "latency": 8.852999687194824, "jitter": 3.253999948501587, "packetLoss": 0.0}, "uplink": {"throughput": 32609168.0, "latency": 13.631999969482422, "jitter": 0.5709999799728394, "packetLoss": 0.0}}, {"downlink": {"throughput": 379424.0, "latency": 6.703999996185303, "jitter": 2.1010000705718994, "packetLoss": 0.0}, "uplink": {"throughput": 96182104.0, "latency": 19.655000686645508, "jitter": 0.7310000061988831, "packetLoss": 0.0}}]
2025-07-31 10:06:10.331: Got DL slice allocation; Slice1=0.2104, Slice2=0.7896
2025-07-31 10:06:10.332: Got UL slice allocation; Slice1=0.2406, Slice2=0.7594
2025-07-31 10:06:12.379: Got KPIs from Qosium: [{"downlink": {"throughput": 91272.0, "latency": 7.944, "jitter": 2.837, "packetLoss": 0.0}, "uplink": {"throughput": 33999376.0, "latency": 13.292, "jitter": 0.509, "packetLoss": 0.0}}, {"downlink": {"throughput": 396256.0, "latency": 6.193, "jitter": 1.596, "packetLoss": 0.0}, "uplink": {"throughput": 102652216.0, "latency": 23.763, "jitter": 0.636, "packetLoss": 0.0}}]
2025-07-31 10:06:12.387: Got DL slice allocation; Slice1=0.2108, Slice2=0.7892
2025-07-31 10:06:12.387: Got UL slice allocation; Slice1=0.2414, Slice2=0.7586
2025-07-31 10:06:14.423: Got KPIs from Qosium: [{"downlink": {"throughput": 94600.0, "latency": 6.899, "jitter": 3.781, "packetLoss": 0.0}, "uplink": {"throughput": 32875504.0, "latency": 13.201, "jitter": 0.6, "packetLoss": 0.0}}, {"downlink": {"throughput": 329280.0, "latency": 5.254, "jitter": 0.767, "packetLoss": 0.0}, "uplink": {"throughput": 92207690.0, "latency": 18.834, "jitter": 0.791, "packetLoss": 0.0}}]
2025-07-31 10:06:14.431: Got DL slice allocation; Slice1=0.2202, Slice2=0.7798
2025-07-31 10:06:14.431: Got UL slice allocation; Slice1=0.2587, Slice2=0.7413
2025-07-31 10:06:16.469: Got KPIs from Qosium: [{"downlink": {"throughput": 90952.0, "latency": 9.643, "jitter": 0.498, "packetLoss": 0.0}, "uplink": {"throughput": 36100336.0, "latency": 15.444, "jitter": 0.665, "packetLoss": 0.0}}, {"downlink": {"throughput": 359456.0, "latency": 8.158, "jitter": 2.436, "packetLoss": 0.0}, "uplink": {"throughput": 100863896.0, "latency": 20.333, "jitter": 0.435, "packetLoss": 0.0}}]
2025-07-31 10:06:16.477: Got DL slice allocation; Slice1=0.209, Slice2=0.791
2025-07-31 10:06:16.478: Got UL slice allocation; Slice1=0.2374, Slice2=0.7626
2025-07-31 10:06:18.518: Got KPIs from Qosium: [{"downlink": {"throughput": 81960.0, "latency": 7.261, "jitter": 4.843, "packetLoss": 0.0}, "uplink": {"throughput": 27760712.0, "latency": 13.278, "jitter": 0.849, "packetLoss": 0.0}}, {"downlink": {"throughput": 209920.0, "latency": 5.986, "jitter": 1.276, "packetLoss": 0.0}, "uplink": {"throughput": 71804010.0, "latency": 19.598475, "jitter": 0.325, "packetLoss": 0.0}}]
2025-07-31 10:06:18.526: Got DL slice allocation; Slice1=0.2328, Slice2=0.7672
2025-07-31 10:06:18.526: Got UL slice allocation; Slice1=0.276, Slice2=0.724
2025-07-31 10:06:20.565: Got KPIs from Qosium: [{"downlink": {"throughput": 96008.0, "latency": 8.338, "jitter": 1.479, "packetLoss": 0.0}, "uplink": {"throughput": 38378912.0, "latency": 12.924, "jitter": 0.51, "packetLoss": 0.0}}, {"downlink": {"throughput": 238208.0, "latency": 5.73, "jitter": 0.331, "packetLoss": 0.0}, "uplink": {"throughput": 82654270.0, "latency": 17.393, "jitter": 0.29, "packetLoss": 0.0}}]
2025-07-31 10:06:20.573: Got DL slice allocation; Slice1=0.2553, Slice2=0.7447
2025-07-31 10:06:20.573: Got UL slice allocation; Slice1=0.3032, Slice2=0.6968
2025-07-31 10:06:22.611: Got KPIs from Qosium: [{"downlink": {"throughput": 82760.0, "latency": 5.112844467163086, "jitter": 0.7710000276565552, "packetLoss": 0.0}, "uplink": {"throughput": 35309000.0, "latency": 12.359155654907227, "jitter": 0.8740000128746033, "packetLoss": 0.0}}, {"downlink": {"throughput": 361952.0, "latency": 7.211999893188477, "jitter": 2.98799991607666, "packetLoss": 0.0}, "uplink": {"throughput": 94518024.0, "latency": 18.535999298095703, "jitter": 0.2720000147819519, "packetLoss": 0.0}}]
2025-07-31 10:06:22.619: Got DL slice allocation; Slice1=0.2082, Slice2=0.7918
2025-07-31 10:06:22.619: Got UL slice allocation; Slice1=0.2351, Slice2=0.7649
2025-07-31 10:06:24.657: Got KPIs from Qosium: [{"downlink": {"throughput": 77352.0, "latency": 6.802999973297119, "jitter": 2.486999988555908, "packetLoss": 0.0}, "uplink": {"throughput": 32704784.0, "latency": 11.128999710083008, "jitter": 1.444000005722046, "packetLoss": 0.0}}, {"downlink": {"throughput": 292064.0, "latency": 8.805000305175781, "jitter": 2.071000099182129, "packetLoss": 0.0}, "uplink": {"throughput": 68964760.0, "latency": 20.525354385375977, "jitter": 0.36500000953674316, "packetLoss": 0.0}}]
2025-07-31 10:06:24.665: Got DL slice allocation; Slice1=0.2246, Slice2=0.7754
2025-07-31 10:06:24.665: Got UL slice allocation; Slice1=0.2652, Slice2=0.7348
2025-07-31 10:06:26.703: Got KPIs from Qosium: [{"downlink": {"throughput": 80776.0, "latency": 6.0002241134643555, "jitter": 0.9110000133514404, "packetLoss": 0.0}, "uplink": {"throughput": 24415312.0, "latency": 12.772775650024414, "jitter": 0.7699999809265137, "packetLoss": 0.0}}, {"downlink": {"throughput": 332640.0, "latency": 8.581000328063965, "jitter": 0.4309999942779541, "packetLoss": 0.0}, "uplink": {"throughput": 81737784.0, "latency": 16.274999618530273, "jitter": 0.27300000190734863, "packetLoss": 0.0}}]
2025-07-31 10:06:26.710: Got DL slice allocation; Slice1=0.2148, Slice2=0.7852
2025-07-31 10:06:26.710: Got UL slice allocation; Slice1=0.2497, Slice2=0.7503
2025-07-31 10:06:28.752: Got KPIs from Qosium: [{"downlink": {"throughput": 82120.0, "latency": 6.045088768005371, "jitter": 1.2450000047683716, "packetLoss": 0.0}, "uplink": {"throughput": 27038336.0, "latency": 12.213911056518555, "jitter": 0.6359999775886536, "packetLoss": 0.0}}, {"downlink": {"throughput": 313920.0, "latency": 6.070000171661377, "jitter": 0.027000000700354576, "packetLoss": 0.0}, "uplink": {"throughput": 93124688.0, "latency": 15.425999641418457, "jitter": 0.2630000114440918, "packetLoss": 0.0}}]
2025-07-31 10:06:28.760: Got DL slice allocation; Slice1=0.2197, Slice2=0.7803
2025-07-31 10:06:28.760: Got UL slice allocation; Slice1=0.2582, Slice2=0.7418
2025-07-31 10:06:30.804: Got KPIs from Qosium: [{"downlink": {"throughput": 83528.0, "latency": 7.9679999351501465, "jitter": 1.180999994277954, "packetLoss": 0.0}, "uplink": {"throughput": 27720320.0, "latency": 13.800999641418457, "jitter": 0.7450000047683716, "packetLoss": 0.0}}, {"downlink": {"throughput": 313312.0, "latency": 7.361000061035156, "jitter": 4.2220001220703125, "packetLoss": 0.0}, "uplink": {"throughput": 87358600.0, "latency": 19.48699951171875, "jitter": 0.718999981880188, "packetLoss": 0.0}}]
2025-07-31 10:06:30.811: Got DL slice allocation; Slice1=0.2079, Slice2=0.7921
2025-07-31 10:06:30.812: Got UL slice allocation; Slice1=0.2346, Slice2=0.7654
2025-07-31 10:06:32.845: Got KPIs from Qosium: [{"downlink": {"throughput": 75112.0, "latency": 6.164999961853027, "jitter": 1.7100000381469727, "packetLoss": 0.0}, "uplink": {"throughput": 28245952.0, "latency": 11.053000450134277, "jitter": 1.597000002861023, "packetLoss": 0.0}}, {"downlink": {"throughput": 293952.0, "latency": 9.600000381469727, "jitter": 2.890000104904175, "packetLoss": 0.0}, "uplink": {"throughput": 79630200.0, "latency": 15.333000183105469, "jitter": 0.29499998688697815, "packetLoss": 0.0}}]
2025-07-31 10:06:32.853: Got DL slice allocation; Slice1=0.2109, Slice2=0.7891
2025-07-31 10:06:32.853: Got UL slice allocation; Slice1=0.2414, Slice2=0.7586
2025-07-31 10:06:34.889: Got KPIs from Qosium: [{"downlink": {"throughput": 83304.0, "latency": 7.2129998207092285, "jitter": 4.934999942779541, "packetLoss": 0.0}, "uplink": {"throughput": 27720224.0, "latency": 14.029999732971191, "jitter": 1.0850000381469727, "packetLoss": 0.0}}, {"downlink": {"throughput": 328480.0, "latency": 5.3379998207092285, "jitter": 0.722000002861023, "packetLoss": 0.0}, "uplink": {"throughput": 93428856.0, "latency": 16.20599937438965, "jitter": 0.25999999046325684, "packetLoss": 0.0}}]
2025-07-31 10:06:34.898: Got DL slice allocation; Slice1=0.2154, Slice2=0.7846
2025-07-31 10:06:34.898: Got UL slice allocation; Slice1=0.2499, Slice2=0.7501
2025-07-31 10:06:36.935: Got KPIs from Qosium: [{"downlink": {"throughput": 86056.0, "latency": 8.170000076293945, "jitter": 3.263000011444092, "packetLoss": 0.0}, "uplink": {"throughput": 29900616.0, "latency": 13.147000312805176, "jitter": 0.6740000247955322, "packetLoss": 0.0}}, {"downlink": {"throughput": 319104.0, "latency": 6.913000106811523, "jitter": 3.36299991607666, "packetLoss": 0.0}, "uplink": {"throughput": 100270424.0, "latency": 21.763999938964844, "jitter": 0.5149999856948853, "packetLoss": 0.0}}]
2025-07-31 10:06:36.943: Got DL slice allocation; Slice1=0.2086, Slice2=0.7914
2025-07-31 10:06:36.943: Got UL slice allocation; Slice1=0.2359, Slice2=0.7641
2025-07-31 10:06:38.982: Got KPIs from Qosium: [{"downlink": {"throughput": 86376.0, "latency": 6.706999778747559, "jitter": 3.367000102996826, "packetLoss": 0.0}, "uplink": {"throughput": 29126344.0, "latency": 14.053000450134277, "jitter": 0.6690000295639038, "packetLoss": 0.0}}, {"downlink": {"throughput": 263552.0, "latency": 6.1570000648498535, "jitter": 1.3919999599456787, "packetLoss": 0.0}, "uplink": {"throughput": 100335384.0, "latency": 23.947999954223633, "jitter": 0.7139999866485596, "packetLoss": 0.0}}]
2025-07-31 10:06:38.990: Got DL slice allocation; Slice1=0.2145, Slice2=0.7855
2025-07-31 10:06:38.990: Got UL slice allocation; Slice1=0.2485, Slice2=0.7515
2025-07-31 10:06:41.044: Got KPIs from Qosium: [{"downlink": {"throughput": 86216.0, "latency": 7.315999984741211, "jitter": 3.9670000076293945, "packetLoss": 0.0}, "uplink": {"throughput": 29376632.0, "latency": 11.378999710083008, "jitter": 1.2050000429153442, "packetLoss": 0.0}}, {"downlink": {"throughput": 330112.0, "latency": 8.081000328063965, "jitter": 1.871000051498413, "packetLoss": 0.0}, "uplink": {"throughput": 99113528.0, "latency": 22.33300018310547, "jitter": 0.6330000162124634, "packetLoss": 0.0}}]
2025-07-31 10:06:41.054: Got DL slice allocation; Slice1=0.2116, Slice2=0.7884
2025-07-31 10:06:41.054: Got UL slice allocation; Slice1=0.2431, Slice2=0.7569
2025-07-31 10:06:43.086: Got KPIs from Qosium: [{"downlink": {"throughput": 79656.0, "latency": 5.895545959472656, "jitter": 0.824999988079071, "packetLoss": 0.0}, "uplink": {"throughput": 28872136.0, "latency": 12.108453750610352, "jitter": 0.7509999871253967, "packetLoss": 0.0}}, {"downlink": {"throughput": 288928.0, "latency": 8.140000343322754, "jitter": 1.4839999675750732, "packetLoss": 0.0}, "uplink": {"throughput": 71189736.0, "latency": 16.014999389648438, "jitter": 0.36500000953674316, "packetLoss": 0.0}}]
2025-07-31 10:06:43.094: Got DL slice allocation; Slice1=0.2142, Slice2=0.7858
2025-07-31 10:06:43.094: Got UL slice allocation; Slice1=0.2483, Slice2=0.7517
2025-07-31 10:06:45.128: Got KPIs from Qosium: [{"downlink": {"throughput": 91336.0, "latency": 9.402999877929688, "jitter": 3.2909998893737793, "packetLoss": 0.0}, "uplink": {"throughput": 30800968.0, "latency": 12.835000038146973, "jitter": 0.7929999828338623, "packetLoss": 0.0}}, {"downlink": {"throughput": 326592.0, "latency": 4.514358043670654, "jitter": 1.5219999551773071, "packetLoss": 0.0}, "uplink": {"throughput": 86822984.0, "latency": 15.733641624450684, "jitter": 0.27900001406669617, "packetLoss": 0.0}}]
2025-07-31 10:06:45.137: Got DL slice allocation; Slice1=0.2178, Slice2=0.7822
2025-07-31 10:06:45.137: Got UL slice allocation; Slice1=0.254, Slice2=0.746
2025-07-31 10:06:47.172: Got KPIs from Qosium: [{"downlink": {"throughput": 83016.0, "latency": 6.486301898956299, "jitter": 1.1629999876022339, "packetLoss": 0.0}, "uplink": {"throughput": 25994680.0, "latency": 12.061697959899902, "jitter": 0.6449999809265137, "packetLoss": 0.0}}, {"downlink": {"throughput": 298528.0, "latency": 7.557000160217285, "jitter": 0.7839999794960022, "packetLoss": 0.0}, "uplink": {"throughput": 98634344.0, "latency": 18.150999069213867, "jitter": 0.28200000524520874, "packetLoss": 0.0}}]
2025-07-31 10:06:47.182: Got DL slice allocation; Slice1=0.2107, Slice2=0.7893
2025-07-31 10:06:47.182: Got UL slice allocation; Slice1=0.2413, Slice2=0.7587
2025-07-31 10:06:49.238: Got KPIs from Qosium: [{"downlink": {"throughput": 89608.0, "latency": 5.498603343963623, "jitter": 3.4760000705718994, "packetLoss": 0.0}, "uplink": {"throughput": 30191696.0, "latency": 13.497396469116211, "jitter": 0.6779999732971191, "packetLoss": 0.0}}, {"downlink": {"throughput": 320576.0, "latency": 9.496999740600586, "jitter": 1.0870000123977661, "packetLoss": 0.0}, "uplink": {"throughput": 104235392.0, "latency": 23.152000427246094, "jitter": 0.46799999475479126, "packetLoss": 0.0}}]
2025-07-31 10:06:49.250: Got DL slice allocation; Slice1=0.2134, Slice2=0.7866
2025-07-31 10:06:49.250: Got UL slice allocation; Slice1=0.2465, Slice2=0.7535
2025-07-31 10:06:51.285: Got KPIs from Qosium: [{"downlink": {"throughput": 72584.0, "latency": 7.956999778747559, "jitter": 1.7410000562667847, "packetLoss": 0.0}, "uplink": {"throughput": 31699432.0, "latency": 15.954000473022461, "jitter": 0.593999981880188, "packetLoss": 0.0}}, {"downlink": {"throughput": 322112.0, "latency": 7.660999774932861, "jitter": 1.5490000247955322, "packetLoss": 0.0}, "uplink": {"throughput": 85186632.0, "latency": 23.454999923706055, "jitter": 1.937999963760376, "packetLoss": 0.0}}]
2025-07-31 10:06:51.293: Got DL slice allocation; Slice1=0.2149, Slice2=0.7851
2025-07-31 10:06:51.293: Got UL slice allocation; Slice1=0.2519, Slice2=0.7481
2025-07-31 10:06:53.329: Got KPIs from Qosium: [{"downlink": {"throughput": 74216.0, "latency": 5.866000175476074, "jitter": 2.140000104904175, "packetLoss": 0.0}, "uplink": {"throughput": 30674136.0, "latency": 13.437999725341797, "jitter": 0.6159999966621399, "packetLoss": 0.0}}, {"downlink": {"throughput": 308864.0, "latency": 9.097999572753906, "jitter": 2.371999979019165, "packetLoss": 0.0}, "uplink": {"throughput": 93517736.0, "latency": 21.15999984741211, "jitter": 0.27000001072883606, "packetLoss": 0.0}}]
2025-07-31 10:06:53.336: Got DL slice allocation; Slice1=0.2087, Slice2=0.7913
2025-07-31 10:06:53.336: Got UL slice allocation; Slice1=0.2361, Slice2=0.7639
"""

def parse_and_prepare_data(log_text):
    """
    Parses the raw log text into a structured pandas DataFrame. This version is
    robust and avoids complex regular expressions.
    """
    records = []
    current_kpi_data = None
    current_timestamp = None

    # Handle potential date inconsistencies in logs
    def robust_strptime(ts_str):
        try:
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            # Fallback for timestamps that might have a different format or be corrupted
            # This is a basic example; more complex logic can be added if needed
            return None

    for line in log_text.strip().split('\n'):
        try:
            timestamp_str = line[:23]
            timestamp = robust_strptime(timestamp_str)
            if timestamp is None: continue

            if "Got KPIs from Qosium:" in line:
                json_part = line.split('Got KPIs from Qosium: ')[1]
                kpis = json.loads(json_part)
                current_kpi_data = kpis
                current_timestamp = timestamp
                
            elif "Got DL slice allocation;" in line:
                parts = line.split(';')[-1].strip().split(',')
                dl_alloc_s1 = float(parts[0].split('=')[1])
                dl_alloc_s2 = float(parts[1].split('=')[1])
                
            elif "Got UL slice allocation;" in line:
                parts = line.split(';')[-1].strip().split(',')
                ul_alloc_s1 = float(parts[0].split('=')[1])
                ul_alloc_s2 = float(parts[1].split('=')[1])
                
                if current_kpi_data and current_timestamp:
                    record = {
                        'timestamp': current_timestamp,
                        'dl_thr_s1': current_kpi_data[0]['downlink']['throughput'],
                        'ul_thr_s1': current_kpi_data[0]['uplink']['throughput'],
                        'dl_thr_s2': current_kpi_data[1]['downlink']['throughput'],
                        'ul_thr_s2': current_kpi_data[1]['uplink']['throughput'],
                        'dl_alloc_s1': dl_alloc_s1,
                        'ul_alloc_s1': ul_alloc_s1,
                    }
                    records.append(record)
                
                current_kpi_data = None
                current_timestamp = None
        except Exception as e:
            print(f"Skipping problematic line: {line.strip()} | Error: {e}")
            continue
            
    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index('timestamp', inplace=True)
    return df

def plot_combined_live_results(df_dict):
    """
    Generates a graph visualizing multiple live network sessions.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    fig.suptitle('Comprehensive Live Network Performance of AI Slicing Agent', fontsize=20, weight='bold')

    colors = {
        'Log 1 (July 30)': ('blue', 'deepskyblue'),
        'Log 2 (July 31)': ('red', 'salmon'),
        'Log 3 (Aug 05)': ('green', 'lime'),
    }

    for log_name, df in df_dict.items():
        if df.empty:
            continue
        
        color_s1, color_s2 = colors[log_name]
        
        # Plot throughputs
        ax1.plot(df.index, df['dl_thr_s1'], label=f'S1 DL Thr ({log_name})', color=color_s1, linewidth=2)
        ax1.plot(df.index, df['dl_thr_s2'], label=f'S2 DL Thr ({log_name})', color=color_s2, linewidth=2, linestyle='--')
        
        # Plot allocations
        ax2.plot(df.index, df['dl_alloc_s1'], label=f'S1 DL Alloc ({log_name})', color=color_s1, marker='o', markersize=5, linestyle='-')
        
    # Formatting for Top Panel (Throughput)
    ax1.set_title('Observed Network Conditions (Downlink Throughput)', fontsize=16)
    ax1.set_yscale('log')
    ax1.set_ylabel('Throughput (log scale, bps)', fontsize=14)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Formatting for Bottom Panel (Allocation)
    ax2.set_title("AI Agent's Allocation Decisions (Downlink)", fontsize=16)
    ax2.set_ylabel('Allocation to Slice 1', fontsize=14)
    ax2.set_ylim(0, 1.0)
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='50/50 Split')
    ax2.legend()
    ax2.grid(True, linestyle='--', linewidth=0.5)

    fig.autofmt_xdate()
    ax2.set_xlabel('Time', fontsize=14)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_filename = "combined_live_performance_all_logs.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*50)
    print("✅ Combined graph saved successfully!")
    print(f"   Location: {os.path.abspath(save_filename)}")
    print("="*50)

    print("\nDisplaying plot window. Close the window to exit the script.")
    plt.show(block=True)

# --- Main execution block ---
if __name__ == "__main__":
    try:
        # I have combined all the log data into one string for simplicity of sharing.
        # In a real script, you might load these from separate files.
        full_log_text = LOG_DATA_1 + "\n" + LOG_DATA_2 + "\n" + LOG_DATA_3

        all_df = parse_and_prepare_data(full_log_text)

        # Split the combined DataFrame back into individual logs based on date
        df_log1 = all_df[all_df.index.day == 30]
        df_log2 = all_df[all_df.index.day == 31]
        df_log3 = all_df[all_df.index.day == 5]
        
        all_logs_dict = {
            'Log 1 (July 30)': df_log1,
            'Log 2 (July 31)': df_log2,
            'Log 3 (Aug 05)': df_log3
        }

        for name, df in all_logs_dict.items():
            print(f"Parsed {len(df)} records from {name}.")
            if df.empty:
                print(f"Warning: {name} could not be parsed or contained no data.")
        
        # Plot all available data
        plot_combined_live_results(all_logs_dict)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nAn error occurred: {e}")
        print("Please ensure pandas and matplotlib are installed (`pip install pandas matplotlib`).")
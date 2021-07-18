# Names for the types of tasks.
#
# @ contactrika
#
import numpy as np


TASK_TYPES = ['HangBag', 'HangCloth', ]
#             'Button', 'Dress', 'Hoop', 'Lasso', 'Mask']


SCENE_INFO = {
    'hang': {
        'entities': {
            'urdf/cuboid.urdf': {
                'basePosition': [0.0, -0.15, 0.20],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 1.0,
            },
            'urdf/hook.urdf': {
                'basePosition': [0.0, (0.3+0.1)/2-0.15, 0.30],
                'baseOrientation': [0, 0, np.pi/2],
                'globalScaling': 1.0,
            },
        },
        'goal_pos_hard': [0.00, 0.03, 0.31],
        'goal_pos_easy': [0.00, 0.13, 0.33],
    },
    'button': {
        'entities': {
            'urdf/torso.urdf': {
                'basePosition': [0.0, 0.0, 0.15],
                'baseOrientation': [0, 0, 0, 1],
                'globalScaling': 1.28,
            },
            'urdf/button_fixed.urdf': {
                'basePosition': [-0.02, 0.13, 0.240],
                'baseOrientation': [0, 0, 0, 1],
                'globalScaling': 1.28,
            },
            'urdf/button_fixed2.urdf': {
                'basePosition': [0.00, 0.13, 0.13],
                'baseOrientation': [0, 0, 0, 1],
                'globalScaling': 1.28,
            },

        },
        'goal_pos_hard': [-0.02, 0.13, 0.250],
        'goal_pos_easy': [-0.02, 0.15, 0.250],
    },
}


DEFORM_INFO = {
    'cloth/ts_apron_twoloops.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos' : [-0.04, 0.35, 0.75],
        'other_anchor_init_pos' : [0.04, 0.35, 0.75],
        'deform_init_pos': [0, 0.40, 0.57],
        'deform_init_ori': [0, 0, np.pi/2],
        'deform_scale': 0.8,
        'deform_elastic_stiffness': 1.0,
        'deform_bending_stiffness': 1.0,
        'deform_anchored_vertex_ids': [
            [1158, 684, 1326, 1325, 1321, 1255, 1250, 683, 1015, 469, 470, 1235,
             1014, 1013, 479, 130, 1159, 1145, 1085, 478, 1087, 143, 131, 1160,
             1083, 542],
            [885, 116, 117, 118, 738, 495, 1210, 884, 1252, 883, 882, 881, 496,
             163, 164, 737, 165, 1290, 1166, 544, 739, 114, 115, 753, 886, 887]
        ],
        'deform_true_loop_vertices': [
            [81, 116, 117, 131, 145, 149, 150, 155, 160, 161, 164,
             168, 176, 299, 375, 377, 480, 483, 492, 497, 500, 501,
             502, 503, 504, 514, 521, 525, 539, 540, 542, 545, 548,
             735, 740, 743, 754, 761, 873, 992, 1019, 1084, 1149, 1159,
             1161, 1167, 1168, 1210, 1255, 1257],
            [51, 53, 57, 68, 157, 162, 170, 177, 179, 181, 182,
             185, 186, 195, 199, 201, 202, 205, 207, 229, 232, 240,
             295, 296, 297, 308, 309, 318, 328, 334, 361, 364, 365,
             367, 383, 392, 402, 409, 411, 414, 508, 510, 511, 527,
             530, 531, 532, 533, 536, 549, 551, 560, 577, 628, 633,
             647, 679, 680, 690, 691, 749, 752, 755, 771, 835, 854,
             856, 857, 859, 860, 867, 871, 872, 986, 988, 989, 990,
             991, 1001, 1008, 1021, 1023, 1152, 1153, 1163, 1164, 1169, 1170,
             1173, 1174, 1175, 1197, 1211, 1228, 1254, 1259, 1260, 1271, 1308,
             1319]
        ],
    },
    'bags/ts_small_bag_resampled.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'deform_init_pos': [0.0, 0.40, 0.57],
        'deform_init_ori': [0, 0, np.pi/2],
        'deform_elastic_stiffness': 1.0,
        'deform_bending_stiffness': 1.0,
        'deform_anchored_vertex_ids': [
            [622, 815, 797, 633, 623, 741, 632, 857, 98, 631, 589, 814, 743,
             588, 8, 742, 587],
            [645, 720, 724, 829, 690, 830, 795, 95, 691, 783, 726, 643, 727,
             841, 644, 699, 722]],
        'deform_true_loop_vertices': [
            [2, 12, 14, 25, 26, 27, 28, 30, 32, 33, 34, 37, 66,
             88, 90, 94, 96, 97, 98, 100, 102, 104, 125, 126, 128, 204,
             250, 297, 301, 337, 339, 378, 382, 510, 525, 528, 529, 530, 532,
             533, 534, 571, 580, 610, 615, 617, 623, 624, 625, 627, 690, 700,
             701, 706, 710, 715, 719, 720, 722, 723, 725, 726, 729, 730, 731,
             732, 734, 736, 739, 740, 741, 743, 744, 749, 751, 753, 755, 756,
             758, 759, 762, 763, 768, 776, 777, 784, 792, 793, 795, 796, 807,
             812, 814, 816, 817, 818, 827, 830, 844, 859]
        ]
    },
}

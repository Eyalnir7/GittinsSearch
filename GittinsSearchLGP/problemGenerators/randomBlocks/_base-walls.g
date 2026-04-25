world { X: [0, 0, .1] }

#thick radius floor and walls

floor (world): { shape: ssBox, Q: [0, 0, -0.05], size: [10, 10, .1, .04], color: [.5 .1], contact: 0 friction: 10, logical: {is_place} }

floor_visual (floor): { shape: ssBox, size: [10, 10, .08, .04], color: [0.6953, 0.515625, .453125] }
        
wall_right (floor): { shape: ssBox, Q: [0, -5, 0.2], size: [10, .2, .3, .04], color: [0.6953, 0.515625, .453125], contact: 1 }
wall_back (floor): { shape: ssBox, Q: [5, 0, 0.2], size: [.2, 10, .3, .04], color: [0.6953, 0.515625, .453125], contact: 1 }
wall_left (floor): { shape: ssBox, Q: [0, 5, 0.2], size: [10, .2, .3, .04], color: [0.6953, 0.515625, .453125], contact: 1 }
wall_front (floor): { shape: ssBox, Q: [-5, 0, 0.2], size: [.2, 10, .3, .04] , color: [0.6953, 0.515625, .453125], contact: 1 }




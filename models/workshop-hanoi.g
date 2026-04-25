Include: 'pandasTableLocal.g'

box1 (table){
    joint:rigid
    shape:ssBox, Q:<t(.6 .3 .25) d(0 0 0 1)> , size:[.3 .3 .4 .02] >, color:[.6 .6 .6], contact 
    friction:.1
    logical={is_heavy, is_box}
}

box2 (table){
    joint:rigid ,
    shape:ssBox, 
Q:[-.6 .3 .25] ,size:[.3 .3 .4 .02] ,
    color:[.6 .6 .6], contact
    friction:.1
    logical={is_heavy, is_box , top_free}
}

block1 (box1){ joint:rigid, shape:ssBox, Q:[0 .0 .25 1 0 0 1], size:[.06 .15 .09 .01], color:[.8 .6 .6], contact }

block2 (block1){ joint:rigid, shape:ssBox, Q:[0 .0 .09 1 0 0 0], size:[.06 .15 .09 .01], color:[.6 .8 .6], contact }

block3 (block2){ joint:rigid, shape:ssBox, Q:[0 .0 .09 1 0 0 0], size:[.06 .15 .09 .01], color:[.6 .6 .8], contact, logical={top_free, object} }


box3 (table){
    joint:rigid
    shape:ssBox,
 Q:[0 .3 .25] ,size:[.3 .3 .4 .02] ,
    color:[.6 .6 .6], contact
    friction:.1
    logical={is_heavy, is_box, top_free}
}






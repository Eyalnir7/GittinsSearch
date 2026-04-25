
floor { shape:ssBox, size:[5., 5., .2, .05], color:[.4,.4,.4], contact=1, logical:{table, object} }

box (floor){ Q:<t(2. 2. .4)> shape:ssBox, size:[.6,.6,.6,.05], color:[.8,.8,.8],
joint:rigid, contact:1, logical:{object, table} }

banana { X:<t(-2. 1. 2.)> shape:ssBox, size:[.1,.1,.3,.02], color:[.9,.8,0], logical:{object} }


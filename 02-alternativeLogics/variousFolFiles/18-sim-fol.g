ANY, QUIT, WAIT, Terminate, INFEASIBLE
moves, touch, stable, dynamic, impulse, goForward

FOL_World{
  hasWait=true
  gamma = 1.
  stepCost = 1.
  timeCost = 0.
}

object
partOf
table
goable

## objects
#hand
#redBall
#blueBall
#bucket


START_STATE {
#    (object hand) (object redBall) (object blueBall) (object bucket)
#    (moves hand)
}
REWARD {}

DecisionRule genTouch{ A, B
    { (moves A) (touch A B)! (touch B A)! (object A) (object B) }
    { (touch A B) (goable) }
}

DecisionRule genStable{ A, B
    { (touch A B) (stable A B)! }
    { (stable ANY B)! (dynamic ANY B)! (stable A B) (goable) }
}

DecisionRule genDynamic{ A, B
    { (dynamic A B)! (table A) (object B)}
    { (stable ANY B)! (dynamic ANY B)! (dynamic A B) (goable) }
}

DecisionRule genImpulse{ A, B
    { (moves A) (touch A B)! (touch B A)! (object A) (object B) }
    { _(touch A B) _(impulse A B) }
}

DecisionRule breakTouch{ A, B
    { (touch A B) }
    { (touch A B)! }
}

DecisionRule go{
    { (goable) }
    { (goForward) (goable)! }
}

Rule { A, B
    { (stable A B) (moves A) }
    { (moves B) }
}

Rule {  A, B
  { (partOf A B) }
  { (moves B) }
}

Rule {  A, B
  { (partOf B A) }
  { (moves B) }
}

ANY, QUIT, WAIT, Terminate, INFEASIBLE
moves, touch, stable, dynOn, impulse, goForward
dynFree

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

DecisionRule briefTouch{ A, B
    { (moves A) (touch A B)! (touch B A)! (object A) (object B) }
    { brief(touch A B) (goable) }
}

DecisionRule genStable{ A, B
    { (touch A B) (stable A B)! }
    { (stable ANY B)! (dynOn ANY B)! (stable A B) (goable) }
}

DecisionRule genDynOn{ A, B
    { (dynOn A B)! (table A) (object B)}
    { (stable ANY B)! (dynOn ANY B)! (dynOn A B) (moves B) (goable) }
}

DecisionRule genDynFree{ A
    { (dynFree A)! (object A)}
    { (stable ANY A)! (dynFree A) (moves A) }
}

DecisionRule genImpulse{ A, B
    { (moves A) (touch A B)! (touch B A)! (object A) (object B) }
    { _(touch A B) _(impulse A B) (moves B) }
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

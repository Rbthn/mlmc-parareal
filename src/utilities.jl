# drop entry from NamedTuple
drop(nt::NamedTuple, key::Symbol) =
    Base.structdiff(nt, NamedTuple{(key,)})

# integer divide, round up
div_up = (x, y) -> ceil(Int, x / y)

# drop entry from NamedTuple
drop(nt::NamedTuple, key::Symbol) =
    Base.structdiff(nt, NamedTuple{(key,)})

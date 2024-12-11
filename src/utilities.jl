function namedtuple_to_dict(vars::NamedTuple)
    return Dict(string(k) => v for (k, v) in pairs(vars))
end

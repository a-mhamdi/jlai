cprintln(args...; kwargs...) = printstyled(IOContext(stdout, :color => true), args..., "\n"; kwargs...)

struct Neg
    fact::Symbol
end

# Convenience so we can write ¬(:flies) or Neg(:flies)
¬(s::Symbol) = Neg(s)

#=
A `Rule` bundles:
- `name`: a label used only for readable trace output (e.g. `"R1"`).
- `premises`: the conjunction of facts that must **all** be in working memory
  (`wm`) for the rule to be eligible to fire.
- `conclusion`: the single fact added to `wm` when the rule fires.

Because `premises` is a plain `Vector{Symbol}`, checking "are all premises known?" is just a set-subset test (`issubset`) against working memory.
=#
        
struct Rule
    name::String                          # human-readable id, e.g. "R1", used only for tracing
    premises::Vector{Union{Symbol, Neg}}  # conjunction of facts required to fire
    conclusion::Symbol                    # fact asserted when the rule fires
end


# THE FORWARD-CHAINING ENGINE
""" Strategy: fire everything that matches. """

holds(p::Symbol, wm::Set{Symbol}) = p in wm
holds(p::Neg,    wm::Set{Symbol}) = !(p.fact in wm)

"""
Repeatedly fire every rule whose premises are satisfied and whose conclusion is not already known, until a full pass adds nothing new.
Mutates and returns `wm`.
"""
function forward_chain!(wm::Set{Symbol}, rules::Vector{Rule}; verbose::Bool=true)
    pass = 1
    changed = true
    while changed
        changed = false
        verbose && cprintln("\n--- PASS $pass ---"; bold=true, underline=true)
        for rule in rules
            # Fire iff every premise is already known AND the conclusion is new.
            if all(p -> holds(p, wm), rule.premises) && !(rule.conclusion in wm)  # O(rules x premises) / pass
                push!(wm, rule.conclusion)
                changed = true  # keep looping: this new fact may unblock other rules
                if verbose
                    cprintln("  Fired $(rule.name): ",
                            join(rule.premises, " \u2227 "),
                            " -> ", rule.conclusion,
                            "   [new fact: $(rule.conclusion)]", color=:green, bold=true)
                end
            end
        end
        pass += 1
    end
    verbose && cprintln("--- Fixpoint reached: no more rules apply ---", color=:red, bold=true)
    return wm
end



# THE BACKWARD-CHAINING ENGINE

"""
**Indexing rules by conclusion:** Backward chaining repeatedly asks, "Which rule(s) conclude *this* goal?" So it pays to index `RULES` by conclusion once, up front, rather than linear-scanning the whole rule base for every sub-goal. `Dict{Symbol, Vector{Rule}}` maps a conclusion to the (order-preserving) list of rules that can prove it.

Group rules by conclusion, preserving the order they appear in RULES.
"""
function rules_by_conclusion(rules::Vector{Rule})
    d = Dict{Symbol,Vector{Rule}}()
    for r in rules
        push!(get!(d, r.conclusion, Rule[]), r)
    end
    return d
end



""" 
            Asking For Primitive _(Askable)_ Facts

A predicate with **no** rule concluding it (e.g. `:has_hair`, `:eats_meat`)
is, by construction, a *primitive fact* the engine cannot derive — it must be
asked. `ask_user` supports two modes:

- **Scripted** (`scripted_answers`): looks up a pre-decided answer, for
  reproducible demos/tests that don't block on stdin.
- **Interactive**: prompts on stdin (`y`/`n`) when no scripted answer exists.

Return true/false for a fact that no rule can derive. If `fact` has a pre-scripted answer (used for reproducible demos / testing), use it;
otherwise prompt interactively on stdin. A scripted answer of `false` represents either an explicit "no" or an "I don\'t know" — for proof
purposes both mean the fact cannot be established as true, exactly as in the hand-worked exercise where "unknown" made a premise fail.
"""

function ask_user(fact::Symbol, scripted_answers::Dict{Symbol,Bool})
    if haskey(scripted_answers, fact)
        ans = scripted_answers[fact]
        println("  Q: $(fact)? (scripted) -> ", ans ? "yes" : "no / unknown")
        return ans
    else
        print("  Q: $(fact)? [y/n] ")
        resp = strip(lowercase(readline()))
        return resp == "y" || resp == "yes"
    end
end


            
"""
`prove_premise!(goal, ...)` is a recursive goal-prover with **memoization** and **backtracking**
"""
function prove_premise!(p::Symbol, rules_idx, known_true, known_false, answers; depth, verbose)
    prove_premise!(p, rules_idx, known_true, known_false, answers; depth=depth, verbose=verbose)
end

function prove_premise!(p::Neg, rules_idx, known_true, known_false, answers; depth, verbose)
    verbose && println("  "^depth, "¬$(p.fact): proving $(p.fact) then negating")
    !prove_premise!(p.fact, rules_idx, known_true, known_false, answers; depth=depth+1, verbose=verbose)
end

"""                                          
Try to prove `goal` using backward chaining:
  1. Return cached result if `goal` was already proven true or false.
  2. If some rule concludes `goal`, try each such rule IN ORDER, recursively proving all of its premises; the first rule whose
     premises all succeed proves the goal (backtrack to the next rule on failure).
  3. If no rule concludes `goal`, it\'s a primitive fact: ask the user.

Mutates `known_true` / `known_false` as a memo cache, so a fact or sub-goal already settled earlier is never re-derived or re-asked.
"""

function backward_chain!(goal::Symbol,
                 rules_idx::Dict{Symbol,Vector{Rule}},
                 known_true::Set{Symbol},
                 known_false::Set{Symbol},
                 answers::Dict{Symbol,Bool};
                 depth::Int=0,
                 verbose::Bool=true)

    indent = "  " ^ depth

    if goal in known_true
        verbose && println("$(indent)$(goal) already known TRUE (cached)")
        return true
    elseif goal in known_false
        verbose && println("$(indent)$(goal) already known FALSE (cached)")
        return false
    end

    candidates = get(rules_idx, goal, Rule[])

    if isempty(candidates)
        # No rule concludes this predicate -> it\'s an askable primitive fact.
        verbose && cprintln("$(indent)$(goal): no rule concludes it -> ask user"; bold=true, color=red)
        result = ask_user(goal, answers)
        push!(result ? known_true : known_false, goal)
        return result
    end

    for rule in candidates
        verbose && println("$(indent)Goal $(goal): trying $(rule.name)  [premises: $(join(rule.premises, ", "))]")
        success = true

        for premise in rule.premises
            if !prove_premise!(premise, rules_idx, known_true, known_false, answers; depth=depth+1, verbose=verbose)
                verbose && cprintln("$(indent)  $(rule.name) FAILS ($(premise) not provable) -> backtrack"; bold=true, color=red)
                success = false
                break
            end
        end
        if success
            push!(known_true, goal)
            verbose && cprintln("$(indent)$(rule.name) succeeds -> $(goal) = TRUE"; bold=true, color=green)
            return true
        end
    end

    # Every candidate rule failed.
    push!(known_false, goal)
    verbose && println("$(indent)No rule proved $(goal) -> $(goal) = FALSE")
    return false
end

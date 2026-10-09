using Documenter, DocumenterCitations
using ProximalAlgorithms, ProximalCore
using Literate

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

src_path = joinpath(@__DIR__, "src/")

literate_directories = joinpath.(src_path, ["guide", "examples"])

for directory in literate_directories
    jl_files = filter(p -> endswith(p, ".jl"), readdir(directory; join = true))
    for src in jl_files
        Literate.markdown(src, directory, documenter = true)
    end
end

makedocs(
    modules = [ProximalAlgorithms, ProximalCore],
    sitename = "ProximalAlgorithms.jl",
    pages = [
        "Home" => "index.md",
        "User guide" => [
            joinpath("guide", "getting_started.md"),
            joinpath("guide", "implemented_algorithms.md"),
            joinpath("guide", "custom_objectives.md"),
            joinpath("guide", "custom_algorithms.md"),
        ],
        "Examples" => [joinpath("examples", "sparse_linear_regression.md")],
        "Bibliography" => "bibliography.md",
    ],
    plugins = [bib],
    checkdocs = :exported,
    # An integration branch collects work in progress; its docs deploy even while incomplete.
    warnonly = Symbol.(split(get(ENV, "DOCUMENTER_WARNONLY", ""), ','; keepempty = false)),
)

# A fork deploys to its own GitHub Pages; its workflow names the branch to deploy as `dev`.
deploydocs(
    repo = "github.com/" * get(ENV, "GITHUB_REPOSITORY", "JuliaFirstOrder/ProximalAlgorithms.jl") * ".git",
    devbranch = get(ENV, "DOCUMENTER_DEVBRANCH", "master"),
)

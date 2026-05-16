using Documenter, ShiftShareIV

makedocs(
    sitename = "ShiftShareIV.jl",
    modules  = [ShiftShareIV],
    pages = [
        "Home"      => "index.md",
        "Vignettes" => [
            "Introduction"              => "vignettes/01_introduction.md",
            "Rotemberg Decomposition"   => "vignettes/02_rotemberg.md",
            "BHJ Shock-Level Inference" => "vignettes/03_bhj.md",
        ],
        "Reference" => "reference.md",
    ],
    warnonly = true,
    remotes  = nothing,
)

deploydocs(
    repo       = "github.com/xiangao/ShiftShareIV.jl.git",
    devbranch  = "main",
    push_preview = false,
)

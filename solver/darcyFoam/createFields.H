#include "Function1.H"

Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
    mesh
);

IOdictionary reservoirProps
(
    IOobject("reservoirProperties", runTime.constant(), mesh,
             IOobject::MUST_READ, IOobject::NO_WRITE)
);

const scalar mu_val = reservoirProps.get<scalar>("mu");

// EVENT SUPPORT: wellRate is now read as a Function1<scalar> (see
// reservoirProperties -- a "table" entry implementing a rate step at
// t=250 days) rather than a plain scalar, so it can change mid-simulation.
// wellRateFunc must stay in scope for the whole run (used every timestep in
// darcyFoam.C's time loop), so it is an autoPtr held at this (main())
// scope rather than a local temporary.
Foam::autoPtr<Foam::Function1<scalar>> wellRateFunc
(
    Foam::Function1<scalar>::New("wellRate", reservoirProps)
);
const scalar wellRate_val = wellRateFunc->value(runTime.value());

// NOTE: "ct" is intentionally no longer read as a scalar here -- Chapter 3/4
// ensembles both write a spatially-resolved constant/ct field (see
// generate_samples_v3.py). If you ever run a case without that file, add
// it back as a lookupOrDefault fallback and build a uniform field from it.

// Spatially-varying K, phi, ct written by generate_samples_v3.py
Info<< "Reading field K\n" << endl;
volScalarField K_field
(
    IOobject("K", runTime.constant(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
    mesh
);

Info<< "Reading field phi\n" << endl;
volScalarField phi_field
(
    IOobject("phi", runTime.constant(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
    mesh
);

Info<< "Reading field ct\n" << endl;
volScalarField ct_field
(
    IOobject("ct", runTime.constant(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
    mesh
);

// Diffusivity K/mu [m2/(Pa.s)-equivalent] -- stays INSIDE the Laplacian.
// This is the only place K appears; unlike the old K_over_mu_over_phiCt
// field, phi and ct are NOT folded in here.
volScalarField K_over_mu
(
    IOobject("K_over_mu", runTime.timeName(), mesh,
             IOobject::NO_READ, IOobject::NO_WRITE),
    K_field * 9.869e-13 / dimensionedScalar("mu", dimDynamicViscosity, mu_val)
);

// Storage coefficient phi*ct -- multiplies ddt(p), stays OUTSIDE the
// Laplacian. This is what makes the discretized equation match
//   phi*ct * dp/dt = div(K/mu * grad(p)) - q                (Eq. 3.1)
// pointwise, even where phi and ct are both heterogeneous.
volScalarField phiCt_field
(
    IOobject("phiCt_field", runTime.timeName(), mesh,
             IOobject::NO_READ, IOobject::NO_WRITE),
    phi_field * ct_field
);

// Well sink field [1/s]. Initialised here at wellRate(t=0); darcyFoam.C's
// time loop overwrites the wellCells entries every step with
// wellRateFunc->value(runTime.value()), so the event step actually takes
// effect during the run rather than only at t=0.
volScalarField wellSink
(
    IOobject("wellSink", runTime.timeName(), mesh,
             IOobject::NO_READ, IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("wellSink", dimensionSet(-1, 1, 1, 0, 0, 0, 0), 0.0)
);

label wellZoneID = mesh.cellZones().findZoneID("wellCells");
if (wellZoneID >= 0)
{
    const labelList& wellCells = mesh.cellZones()[wellZoneID];
    forAll(wellCells, i)
    {
        wellSink[wellCells[i]] = wellRate_val;
    }
}

Info<< "Well sink rate at t=0: " << wellRate_val << " [1/s] "
    << "(time-varying -- see reservoirProperties for the event schedule)" << nl
    << "K range: "   << min(K_field).value()   << " to " << max(K_field).value()   << " D"  << nl
    << "phi range: " << min(phi_field).value() << " to " << max(phi_field).value() << nl
    << "ct range: "  << min(ct_field).value()  << " to " << max(ct_field).value()  << " 1/Pa" << nl
    << endl;

K_over_mu.correctBoundaryConditions();
phiCt_field.correctBoundaryConditions();
wellSink.correctBoundaryConditions();

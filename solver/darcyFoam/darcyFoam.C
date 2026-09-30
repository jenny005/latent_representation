#include "fvCFD.H"

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createFields.H"

    Info<< "\nStarting time loop\n" << endl;

    // One-shot flag so the event is logged exactly once when it fires,
    // rather than printing every timestep.
    bool eventLogged = false;
    scalar wellRate_prev = wellRate_val;

    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // EVENT SUPPORT: re-evaluate the (possibly time-varying) well sink
        // coefficient every timestep and rewrite it into wellCells. Outside
        // the event window this is a no-op (same value as before); at the
        // step defined in reservoirProperties, wellSink changes here.
        const scalar wellRate_t = wellRateFunc->value(runTime.value());
        if (wellZoneID >= 0)
        {
            const labelList& wellCells = mesh.cellZones()[wellZoneID];
            forAll(wellCells, i)
            {
                wellSink[wellCells[i]] = wellRate_t;
            }
        }
        if (!eventLogged && mag(wellRate_t - wellRate_prev) > SMALL)
        {
            Info<< ">>> EVENT: wellRate changed " << wellRate_prev
                << " -> " << wellRate_t << " [1/s] at t=" << runTime.timeName()
                << " <<<" << nl << endl;
            eventLogged = true;
        }
        wellRate_prev = wellRate_t;

        // Equation (Eq. 3.1): phi*ct * dp/dt = div(K/mu * grad(p)) - wellRate(t)*p
        // phi*ct multiplies ddt(p) directly (phiCt_field), K/mu is the sole
        // diffusivity inside the Laplacian (K_over_mu) -- no lumping of
        // phi/ct into the diffusion operator, so this holds pointwise even
        // where phi and ct are both spatially heterogeneous. wellSink is
        // now time-varying (see EVENT SUPPORT above) but still enters the
        // equation as a linear sink term, same as before.
        fvScalarMatrix pEqn
        (
            fvm::ddt(phiCt_field, p)
          - fvm::laplacian(K_over_mu, p)
          + fvm::Sp(wellSink, p)
        );

        pEqn.solve();

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;
    return 0;
}

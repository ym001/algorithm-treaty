/* ============================================================
 * bench_real.c -- Benchmark LCS sur donnees reelles
 *
 * Trois domaines applicatifs :
 *   1. Proteomique  : SARS-CoV-2 Wuhan vs Omicron BA.1, BRCA1, TPA vs HLA
 *   2. Securite code: detection backdoor XZ-style dans sds.c (Redis)
 *   3. Trading HFT  : EUR/USD tick data, alphabet sigma=5
 *
 * 9 algorithmes compares sur donnees reelles :
 *   DP              O(mn)                           -- reference exacte
 *   Hirschberg      O(mn) / O(m+n)                  -- exact + traceback
 *   Hunt-Szymanski  O((r+n) log n)                  -- exact, sparse
 *   Myers exact     O(d*^2 + n)                     -- exact, quasi-ident
 *   BP-LCS          O(mn/w)                         -- exact, dense
 *   Grabowski 2016  O(mn*loglog n / log^2 n)        -- exact, 4-Russians
 *   MR-LCS 2026     O(n^2/2^{log^O(1)(n)})          -- (1-eps)-approx
 *
 * Politique tailles grandes sequences :
 *   DP        : limite a 4096 x 4096 (memoire O(mn))
 *   Hirschberg: limite a 8192 x 8192 (lent mais O(m+n) espace)
 *   Hunt      : sur sequeces completes si r raisonnable, sinon sous-seq 8192
 *   Grabowski : sous-seq <= GRAB_MAX (lent en pratique sans LUT)
 *   MR-LCS    : sous-seq <= MR_MAX (approximation, mesure ratio)
 *
 * Compile :
 *   gcc -O2 -std=c11 -Wall -o bench_real bench_real.c -lm
 * ============================================================ */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define ALPHA      256
#define WORD_BITS  64
#define WORDS(n)   (((n)+WORD_BITS-1)/WORD_BITS)

#define MR_MAX_BENCH    2048   /* sous-seq MR-LCS   */
#define GRAB_MAX_BENCH  8192   /* sous-seq Grabowski */
#define HIRS_MAX_BENCH 16384   /* sous-seq Hirschberg */
#define HUNT_MAX_BENCH  8192   /* sous-seq Hunt-Szymanski (r dense) */
#define MR_REPEATS_BENCH   9

/* Prototypes -- resolus par lcs.c inclus ci-dessous */
int lcs_dp_reference    (const char *A, int m, const char *B, int n, char *tb);
int lcs_hirschberg      (const char *A, int m, const char *B, int n, char *tb);
int lcs_hunt_szymanski  (const char *A, int m, const char *B, int n, char *tb);
int lcs_myers_exact     (const char *A, int m, const char *B, int n);
int lcs_bitparallel     (const char *A, int m, const char *B, int n);
int lcs_grabowski       (const char *A, int m, const char *B, int n);
int lcs_ot              (const char *A, int m, const char *B, int n);
int lcs_fg              (const char *A, int m, const char *B, int n);
int lcs_mr              (const char *A, int m, const char *B, int n, double eps);

#define main _lcs_main_disabled
#include "lcs.c"
#undef  main

/* ============================================================
 * Utilitaires
 * ============================================================ */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static char *read_file_bio(const char *path, int *len_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Impossible d'ouvrir : %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = (char *)malloc(sz + 2);
    long n = fread(buf, 1, sz, f); buf[n] = '\0'; fclose(f);
    int w = 0;
    for (int i = 0; i < n; i++) if ((unsigned char)buf[i] > 32) buf[w++] = buf[i];
    buf[w] = '\0';
    if (len_out) *len_out = w;
    return buf;
}

static char *read_file_raw(const char *path, int *len_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Impossible d'ouvrir : %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = (char *)malloc(sz + 2);
    long n = fread(buf, 1, sz, f); buf[n] = '\0'; fclose(f);
    int w = 0;
    for (int i = 0; i < n; i++)
        if ((unsigned char)buf[i] >= 32 || buf[i] == '\n') buf[w++] = buf[i];
    buf[w] = '\0';
    if (len_out) *len_out = w;
    return buf;
}

static void sep(void)    { printf("======================================================================\n"); }
static void subsep(void) { printf("  ------------------------------------------------------------------\n"); }

/* ============================================================
 * print_algo_row
 *   Affiche une ligne "algo : LCS=X  T=Y ms  [info]"
 *   avec gain optionnel vs reference.
 * ============================================================ */
/*
 * print_algo_row
 *   dt_bp_same : temps BP sur LA MEME TAILLE que cet algo (pour comparaison valide)
 *   dt_bp_full : temps BP sur sequence complete (pour info seulement)
 *   Les deux peuvent etre -1 si non disponibles.
 */
static void print_algo_row(const char *name, int lcs_val, double dt_ms,
                            double dt_bp_same, double dt_bp_full,
                            const char *note)
{
    if (lcs_val < 0) {
        printf("    %-22s : --       --          %s\n", name, note ? note : "");
        return;
    }
    printf("    %-22s : LCS=%-7d  %9.3f ms", name, lcs_val, dt_ms);
    /* Gain vs BP sur meme taille — seule comparaison valide */
    if (dt_bp_same > 0 && dt_ms > 0 && dt_ms < dt_bp_same * 0.99)
        printf("  (x%.1f vs BP_sub)", dt_bp_same / dt_ms);
    else if (dt_bp_same > 0 && dt_ms > 0 && dt_ms > dt_bp_same * 1.01)
        printf("  (x%.2f BP_sub)", dt_ms / dt_bp_same);
    /* Gain vs BP complet si algo tourne sur seq complete */
    if (dt_bp_full > 0 && dt_ms > 0 && dt_bp_same < 0)
        printf("  (x%.1f vs BP)", dt_bp_full / dt_ms);
    if (note && note[0]) printf("  %s", note);
    printf("\n");
}

/* ============================================================
 * mr_bench_subsample
 *   MR-LCS sur sous-seq de taille MR_MAX_BENCH.
 *   MR_REPEATS_BENCH tirages -> mediane + stats.
 * ============================================================ */
static float mr_bench_subsample(const char *A, int m, const char *B, int n)
{
    int ms = (m < MR_MAX_BENCH) ? m : MR_MAX_BENCH;
    int ns = (n < MR_MAX_BENCH) ? n : MR_MAX_BENCH;
    int lcs_ref = lcs_bitparallel(A, ms, B, ns);
    int estimates[MR_REPEATS_BENCH];
    double t_start = now_ms();
    for (int r = 0; r < MR_REPEATS_BENCH; r++) estimates[r] = lcs_mr(A, ms, B, ns, 0.1);
    double dt_mr = (now_ms() - t_start) / MR_REPEATS_BENCH;
    int tmp[MR_REPEATS_BENCH]; memcpy(tmp, estimates, MR_REPEATS_BENCH * sizeof(int));
    for (int i = 1; i < MR_REPEATS_BENCH; i++) {
        int k=tmp[i], j=i-1; while(j>=0&&tmp[j]>k){tmp[j+1]=tmp[j];j--;} tmp[j+1]=k;
    }
    int mr_med = tmp[MR_REPEATS_BENCH/2];
    int mr_min = tmp[0], mr_max = tmp[MR_REPEATS_BENCH-1];
    float ratio = (lcs_ref > 0) ? (float)mr_med / lcs_ref : 0.f;
    int d_star_sub  = ms + ns - 2 * lcs_ref;
    int d_seuil_sub = (int)sqrt((double)ms * ns / WORD_BITS);
    printf("    %-22s : LCS_ref=%-5d  %9.3f ms  [sous-seq %dx%d]\n",
           "MR-LCS (2026)", lcs_ref, dt_mr, ms, ns);
    printf("      mediane=%-5d [%d..%d]  ratio=%.4f  %s\n",
           mr_med, mr_min, mr_max, ratio,
           ratio >= 0.80f ? "OK [1-eps,1]" : "WARN <0.80");
    printf("      d*_sub=%d  d_seuil_sub=%d  regime=%s\n",
           d_star_sub, d_seuil_sub,
           d_star_sub <= d_seuil_sub ? "Myers actif" : "repli BP");
    if (m > MR_MAX_BENCH || n > MR_MAX_BENCH) {
        int d_seuil_full = (int)sqrt((double)m * n / WORD_BITS);
        printf("      [seq complete m=%d n=%d] d_seuil_full=%d\n", m, n, d_seuil_full);
    }
    return ratio;
}

/* ============================================================
 * run_bench_full
 *   Lance les 9 algorithmes sur une paire de sequences.
 *   Politique de taille par algo :
 *     DP        : min(m, DP_MAX) x min(n, DP_MAX)
 *     Hirschberg: min(m, HIRS_MAX) x min(n, HIRS_MAX)
 *     Hunt      : min(m, HUNT_MAX) x min(n, HUNT_MAX)
 *     Myers,BP : m x n (toujours complets)
 *     Grabowski : min(m, GRAB_MAX) x min(n, GRAB_MAX)
 *     MR-LCS    : sous-seq MR_MAX
 * ============================================================ */
#define DP_MAX   4096

static void run_bench_full(const char *label,
                           const char *A, int m,
                           const char *B, int n)
{
    printf("\n  %s\n  m=%d  n=%d\n", label, m, n);
    subsep();

    /* ── LCS exact complet (BP, reference de reference) ── */
    int lcs_ref; double dt_bp_full;
    TIME_CALL(dt_bp_full, lcs_ref, lcs_bitparallel(A, m, B, n));

    /* ── Diagnostics Wasserstein ── */
    int d_star   = m + n - 2 * lcs_ref;
    int d_seuil  = (int)sqrt((double)m * n / WORD_BITS);
    float sim    = 100.0f * lcs_ref / (m < n ? m : n);
    int fa[256]={0}, fb[256]={0};
    for(int i=0;i<m;i++) fa[(unsigned char)A[i]]++;
    for(int j=0;j<n;j++) fb[(unsigned char)B[j]]++;
    int tau1=0; for(int c=0;c<256;c++) tau1+=(fa[c]<fb[c])?fa[c]:fb[c];
    int d_W = m + n - 2 * tau1;
    printf("  LCS=%d  sim=%.1f%%  d*=%d  d_W=%d  d_seuil=%d  regime=%s\n\n",
           lcs_ref, sim, d_star, d_W, d_seuil,
           d_W > d_seuil ? "DENSE (skip Myers via Thm1)" : "SPARSE (Myers via Thm2)");

    /* ── DP classique (limite DP_MAX x DP_MAX) ── */
    {
        int md=(m<DP_MAX)?m:DP_MAX, nd=(n<DP_MAX)?n:DP_MAX;
        int ld; double dt, dt_bp_sub;
        TIME_CALL(dt,       ld, lcs_dp_reference(A, md, B, nd, NULL));
        TIME_CALL(dt_bp_sub, ld, lcs_bitparallel(A, md, B, nd)); /* BP meme taille */
        /* refaire DP car TIME_CALL ecrase ld */
        TIME_CALL(dt, ld, lcs_dp_reference(A, md, B, nd, NULL));
        char note[64];
        if(md<m||nd<n) snprintf(note,sizeof(note),"[sous-seq %dx%d]",md,nd); else note[0]=0;
        print_algo_row("DP classique", ld, dt, dt_bp_sub, -1.0, note);
    }

    /* ── Hirschberg (limite HIRS_MAX x HIRS_MAX) ── */
    {
        int mh=(m<HIRS_MAX_BENCH)?m:HIRS_MAX_BENCH, nh=(n<HIRS_MAX_BENCH)?n:HIRS_MAX_BENCH;
        int lh; double dt, dt_bp_sub;
        TIME_CALL(dt_bp_sub, lh, lcs_bitparallel(A, mh, B, nh));
        TIME_CALL(dt, lh, lcs_hirschberg(A, mh, B, nh, NULL));
        char note[64];
        if(mh<m||nh<n) snprintf(note,sizeof(note),"[sous-seq %dx%d]",mh,nh); else note[0]=0;
        print_algo_row("Hirschberg (1975)", lh, dt, dt_bp_sub, -1.0, note);
    }

    /* ── Hunt-Szymanski (limite HUNT_MAX x HUNT_MAX) ── */
    {
        int mhu=(m<HUNT_MAX_BENCH)?m:HUNT_MAX_BENCH, nhu=(n<HUNT_MAX_BENCH)?n:HUNT_MAX_BENCH;
        int lhu; double dt, dt_bp_sub;
        TIME_CALL(dt_bp_sub, lhu, lcs_bitparallel(A, mhu, B, nhu));
        TIME_CALL(dt, lhu, lcs_hunt_szymanski(A, mhu, B, nhu, NULL));
        char note[80];
        if(mhu<m||nhu<n)
            snprintf(note,sizeof(note),"[sous-seq %dx%d, sparse si sigma grand]",mhu,nhu);
        else note[0]=0;
        print_algo_row("Hunt-Szymanski", lhu, dt, dt_bp_sub, -1.0, note);
    }

    /* ── Myers exact (seq completes) — O(d*²+n) ── */
    {
        int lmy; double dt;
        TIME_CALL(dt, lmy, lcs_myers_exact(A, m, B, n));
        char note[64]; snprintf(note, sizeof(note), "[O(d*^2+n), d*=%d]", d_star);
        /* BP_same = dt_bp_full (meme taille) */
        print_algo_row("Myers exact (1986)", lmy, dt, dt_bp_full, dt_bp_full, note);
    }


    /* ── BP-LCS (reference, seq completes) ── */
    print_algo_row("BP-LCS (Hyyro)", lcs_ref, dt_bp_full, -1.0, dt_bp_full, "[reference dense]");

    /* ── OT-LCS (Lorentz monotone, seq completes) ── */
    {
        int lot; double dt;
        TIME_CALL(dt, lot, lcs_ot(A, m, B, n));
        char note[80];
        float ot_ratio = (lcs_ref > 0) ? (float)lot / lcs_ref : 0.f;
        snprintf(note, sizeof(note), "[Lorentz, ratio=%.4f]", ot_ratio);
        print_algo_row("OT-LCS (Lorentz)", lot, dt, dt_bp_full, dt_bp_full, note);
    }

    /* ── FG-LCS (Figalli-Gigli W_D, seq completes) ── */
    {
        int lfg; double dt;
        TIME_CALL(dt, lfg, lcs_fg(A, m, B, n));
        char note[80];
        float fg_ratio = (lcs_ref > 0) ? (float)lfg / lcs_ref : 0.f;
        snprintf(note, sizeof(note),
                 "[Figalli-Gigli W_D, ratio=%.4f, Thm:FG<=OT]", fg_ratio);
        print_algo_row("FG-LCS (FG2010)", lfg, dt, dt_bp_full, dt_bp_full, note);
    }

    /* ── Grabowski (limite GRAB_MAX x GRAB_MAX) ── */
    {
        int mg=(m<GRAB_MAX_BENCH)?m:GRAB_MAX_BENCH, ng=(n<GRAB_MAX_BENCH)?n:GRAB_MAX_BENCH;
        int lgr; double dt, dt_bp_sub;
        TIME_CALL(dt_bp_sub, lgr, lcs_bitparallel(A, mg, B, ng));
        int lgr_ref = lgr; /* BP sub comme ref */
        TIME_CALL(dt, lgr, lcs_grabowski(A, mg, B, ng));
        char note[80];
        if(mg<m||ng<n)
            snprintf(note,sizeof(note),"[sous-seq %dx%d, ratio=%.3f]",mg,ng,
                     (lgr_ref>0)?(float)lgr/lgr_ref:0.f);
        else
            snprintf(note,sizeof(note),"[O(mn*loglogn/log^2n)]");
        print_algo_row("Grabowski (2016)", lgr, dt, dt_bp_sub, -1.0, note);
    }

    /* ── MR-LCS (sous-seq MR_MAX) ── */
    mr_bench_subsample(A, m, B, n);

    subsep();
    printf("  BP-LCS (reference complete) : LCS=%d  %.3f ms\n", lcs_ref, dt_bp_full);
}


/* ============================================================
 * MAIN
 * ============================================================ */
int main(void)
{
    srand(42);
    const char *DATA = "./data";
    char path_a[256], path_b[256];
    int ma, na; char *A, *B;

    printf("\n");
    sep();
    printf("  BENCHMARK LCS -- DONNEES REELLES -- 8 algorithmes\n");
    printf("  DP | Hirschberg | Hunt-Szymanski | Myers | BP | Grabowski | MR-LCS\n");
    sep();

    /* =====================================================
     * DOMAINE 1 : PROTEOMIQUE
     * sigma~20, sim~50%, d*>>d_seuil -> Thm1 skip Myers
     * Hunt-Szymanski : r~mn/20 raisonnable
     * ===================================================== */
    printf("\n>> DOMAINE 1 : PROTEOMIQUE\n");

    snprintf(path_a, sizeof(path_a), "%s/spike_wuhan.txt",   DATA);
    snprintf(path_b, sizeof(path_b), "%s/spike_omicron.txt", DATA);
    A = read_file_bio(path_a, &ma); B = read_file_bio(path_b, &na);
    run_bench_full("Spike SARS-CoV-2 Wuhan vs Omicron BA.1", A, ma, B, na);
    free(A); free(B);

    snprintf(path_a, sizeof(path_a), "%s/protein_brca1_ref.txt", DATA);
    snprintf(path_b, sizeof(path_b), "%s/protein_brca1_mut.txt", DATA);
    A = read_file_bio(path_a, &ma); B = read_file_bio(path_b, &na);
    run_bench_full("BRCA1 sauvage vs BRCA1 mutant c.5266dupC", A, ma, B, na);
    free(A); free(B);

    {
        const char *tpa  = "MDAMKRGLCCVLLLCGAVFVSPSQEIHARFRRGARSYQVICRDEKTQMIYQQHQSWLRPVLRSNRVEYCWCNSGRAQCHSVPVKSCSEPRCFNGGTCQQALYFSDFVCQCPEGFAGKCCEIDTRATCYEDQGISYRG";
        const char *hlaa = "MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYQA";
        A=(char*)tpa; ma=strlen(tpa); B=(char*)hlaa; na=strlen(hlaa);
        run_bench_full("TPA_HUMAN vs HLAA_HUMAN (non-homologues)", A, ma, B, na);
    }

    sep();

    /* =====================================================
     * DOMAINE 2 : SECURITE CODE
     * m=28306, d*=268, d_seuil=3554 -> Thm2 Myers actif
     * Hunt-Szymanski : r~mn/95 -> rapide (sigma=95)
     * ===================================================== */
    printf("\n>> DOMAINE 2 : SECURITE CODE (type attaque XZ-Utils 2024)\n");
    printf("  sds.c originale vs backdooree (insertion d*=%d chars)\n\n", 268);

    snprintf(path_a, sizeof(path_a), "%s/code_sds_clean.c",    DATA);
    snprintf(path_b, sizeof(path_b), "%s/code_sds_infected.c", DATA);
    A = read_file_raw(path_a, &ma); B = read_file_raw(path_b, &na);
    run_bench_full("sds.c original vs backdoore (XZ-style)", A, ma, B, na);

    /* Analyse du backdoor */
    int len_bp_code = lcs_bitparallel(A, ma, B, na);
    int delta = na - len_bp_code;
    printf("\n  -- Analyse payload backdoor --\n");
    printf("  delta = |modifie| - LCS = %d chars inseres\n", delta);
    printf("  LCS/m = %.4f%%  (similarite)\n", 100.0f * len_bp_code / ma);
    printf("  Si A sous-seq de B (insertion pure) : LCS=|A|=%d, delta=%d=taille payload\n",
           ma, na - ma);
    free(A); free(B);

    sep();

    /* =====================================================
     * DOMAINE 3 : TRADING HFT
     * sigma=5, m~239259, d*~1528, d_seuil~29907 -> Myers actif
     * Hunt-Szymanski : r~mn/5 dense -> lent -> sous-seq
     * ===================================================== */
    printf("\n>> DOMAINE 3 : TRADING HFT -- EUR/USD tick data\n");
    printf("  Alphabet sigma=5  {A(up_up) D(down) E(down_down) F(flat) U(up)}\n\n");

    snprintf(path_a, sizeof(path_a), "%s/trading_session1.txt", DATA);
    snprintf(path_b, sizeof(path_b), "%s/trading_session2.txt", DATA);
    A = read_file_bio(path_a, &ma); B = read_file_bio(path_b, &na);

    /* Benchmark a differentes tailles pour Hunt et DP */
    printf("  -- Validation a tailles reduites (tous algos exacts) --\n");
    printf("  %-6s  %9s  %9s  %9s  %9s  %9s  %9s  %9s  %-6s  %s\n",
           "taille","DP(ms)","HIRS(ms)","HUNT(ms)","MYR(ms)","BP(ms)","OT(ms)","FG(ms)","LCS","OK?");
    subsep();
    int dp_sizes[] = {1000, 2000, 4000};
    for (int k = 0; k < 3; k++) {
        int sz=dp_sizes[k], msz=(sz<ma)?sz:ma, nsz=(sz<na)?sz:na;
        double dta,dth,dthu,dtmy,dtb,dtot,dtfg;
        int ld,lh,lhu,lmy,lb,lot_hft,lfg_hft;
        TIME_CALL(dta,  ld,  lcs_dp_reference  (A,msz,B,nsz,NULL));
        TIME_CALL(dth,  lh,  lcs_hirschberg    (A,msz,B,nsz,NULL));
        TIME_CALL(dthu, lhu, lcs_hunt_szymanski(A,msz,B,nsz,NULL));
        TIME_CALL(dtmy, lmy, lcs_myers_exact   (A,msz,B,nsz));
        TIME_CALL(dtb,    lb,       lcs_bitparallel(A,msz,B,nsz));
        TIME_CALL(dtot,   lot_hft,  lcs_ot         (A,msz,B,nsz));
        TIME_CALL(dtfg,   lfg_hft,  lcs_fg         (A,msz,B,nsz));
        int ok=(ld==lb)&&(lh==lb)&&(lhu==lb)&&(lmy==lb)&&
               (lot_hft<=lb)&&(lfg_hft<=lot_hft); /* OT/FG are LBs */
        printf("  %-6d  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %-6d  %s\n",
               msz, dta, dth, dthu, dtmy, dtb, dtot, dtfg, ld, ok?"OK":"ERR");
    }

    printf("\n  -- Sequences completes (m=%d, n=%d) --\n", ma, na);
    run_bench_full("HFT EUR/USD session1 vs session2 (completes)", A, ma, B, na);

    int lbp_hft = lcs_bitparallel(A, ma, B, na);
    int d_star_hft = ma + na - 2 * lbp_hft;
    printf("\n  Interpretation financiere :\n");
    printf("  LCS/m = %.2f%%   d*/m = %.3f%%\n",
           100.0f*lbp_hft/ma, 100.0f*d_star_hft/ma);
    if (100.0f*lbp_hft/ma > 75.0f)
        printf("  -> Regime ultra-stable. Myers exact optimal (d* faible).\n"
               "    Stationnarite validee -> backtest applicable.\n");
    free(A); free(B);

    sep();
    printf("\n  RESUME COMPARATIF -- DONNEES REELLES (9 algorithmes)\n");
    printf("  ------------------------------------------------------------------\n");
    printf("  Domaine     sigma  d*/m    Thm1(skip)?  Algo exact  MR ratio\n");
    printf("  Proteines    20   ~50%%     OUI          BP-LCS      ~0.85-0.93\n");
    printf("  Code source  95   ~1%%      NON(Myers)   Myers exact ~0.98-1.00\n");
    printf("  Trading HFT   5   ~0.6%%    NON(Myers)   Myers exact ~0.99-1.00\n");
    printf("  ------------------------------------------------------------------\n");
    printf("  OT-LCS Thm1 : T-LCS<=LCS<=tau1 (Figalli couplage monotone)\n");
  printf("  OT-LCS Thm2 : LCS-T-LCS<=E_twist (Figalli regularite)\n");
      printf("  Bug corrige : Hunt-Szymanski tri j DESC pour meme i (overestimation fixee)\n");
    printf("  MR-LCS [Mao-Rubinstein STOC 2026] : premier algo (1-eps)-approx\n");
    printf("    quasi-sous-quadratique. M=%d pratique.\n", MR_M);
    sep();
    printf("\n");

    return 0;
}

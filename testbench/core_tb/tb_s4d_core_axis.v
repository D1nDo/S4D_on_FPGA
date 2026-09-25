`timescale 1ns / 1ps

//=====================================================================
// tb_s4d_core_axis -- testbench di s4d_core_axis
//
//   Manda al core lo stimolo di stim_2x8x128.mem (2 sequenze da 8 passi
//   temporali x 128 canali) e confronta ogni campione in uscita con
//   expected_2x8x128.mem, prodotto da core_model.py.
//
//   Il core ha TRE interfacce a flusso:
//     s_axis_enc   ingresso: i campioni della sequenza (dall'encoder)
//     s_axis_w     ingresso: i pesi dei 6 layer, che il core chiede uno
//                  alla volta mentre elabora (in scheda arrivano dal DMA)
//     m_axis_out   uscita: la sequenza dopo i 6 layer (verso il decoder)
//   I pesi sono quelli veri della rete addestrata (w_stream.mem) e vanno
//   rimandati da capo per ogni sequenza.
//
//   PARAMETRI DEL MODULO IN QUESTA PROVA
//     F_SSM = 9                valore del progetto ECG (block design)
//     L = 8, LOG2_L = 3        RIDOTTI: in scheda L = 1024, LOG2_L = 10
//   L e' ridotto perche' una sequenza da 1024 passi richiede circa
//   4 milioni di cicli di clock: troppi per una prova da lanciare spesso.
//   L cambia solo la lunghezza della sequenza (memoria e contatori), non
//   i calcoli di un layer.
//
//   Lo stesso stimolo viene mandato DUE volte, con un reset in mezzo:
//     fase 1  alla massima velocita': nessuna pausa sui due ingressi,
//             uscita sempre accettata
//     fase 2  con pause casuali sui campioni e sui pesi (tvalid a 0) e
//             stalli casuali in uscita (tready a 0)
//   I campioni attesi sono gli stessi nelle due fasi.
//
//   Il testbench si giudica da solo: l'ultima riga stampata e'
//   "PASS" oppure "FAIL" con il numero di errori.
//
//   File necessari nella cartella da cui gira il simulatore:
//     stim_2x8x128.mem  expected_2x8x128.mem  w_stream.mem
//     gelu_lut.mem  sigmoid_lut.mem
//   Ordine per rigenerarli:
//     weight_gen.py, gen_core_stimulus.py, core_model.py
//
//   Sorgenti del modulo (cartella rtl/ del repository):
//     s4d_core_axis.v  s4d_wload_axis.v  s4d_seqbuf.v  s4d_layer.v  s4d_affine.v
//     s4d_biquad_bank.v  s4d_biquad_unit.v  s4d_pwl.v  s4d_mixing_glu.v
//=====================================================================

module tb_s4d_core_axis;

    //=================================================================
    // Parametri che si possono cambiare senza toccare il resto
    //=================================================================
    parameter         STIM_FILE            = "stim_2x8x128.mem";       // campioni da mandare
    parameter         WEIGHTS_FILE         = "w_stream.mem";           // pesi dei 6 layer
    parameter         EXPECTED_FILE        = "expected_2x8x128.mem";   // campioni attesi in uscita
    parameter [4:0]   EXPECTED_STATUS      = 5'b00000;   // status_err atteso a fine fase (lo stampa core_model.py)
    parameter integer RANDOM_SEED          = 1;     // seme dei numeri casuali della fase 2
    parameter integer INPUT_PAUSE_PERCENT  = 30;    // fase 2: probabilita' (0..99) di un ciclo di pausa sui campioni
    parameter integer WEIGHT_PAUSE_PERCENT = 30;    // fase 2: probabilita' (0..99) di un ciclo di pausa sui pesi
    parameter integer OUTPUT_STALL_PERCENT = 30;    // fase 2: probabilita' (0..99) di un ciclo di stallo in uscita

    //=================================================================
    // Parametri con cui viene istanziato il modulo
    //=================================================================
    localparam integer H      = 128;   // canali per passo temporale
    localparam integer L      = 8;     // passi temporali per sequenza (ridotto: in scheda 1024)
    localparam integer LOG2_L = 3;     // (ridotto: in scheda 10)
    localparam integer F_SSM  = 9;     // bit frazionari dell'uscita dei filtri (progetto ECG)
    localparam integer NLAYER = 6;     // layer (default del modulo)
    localparam integer W_ACT  = 16;    // bit di un campione
    localparam integer AXI_DW = 64;    // bit di una parola del flusso dei pesi

    //=================================================================
    // Costanti del testbench
    //=================================================================
    localparam integer CLOCK_PERIOD_NS  = 10;        // periodo del clock: 10 ns = 100 MHz
    localparam integer RESET_CYCLES     = 5;         // cicli di clock in cui il reset resta attivo
    localparam integer CYCLES_AFTER_END = 2000;      // cicli di attesa dopo l'ultimo campione, per scoprire campioni in piu'
    localparam integer TIMEOUT_CYCLES   = 1000000;   // durata massima di tutta la simulazione
    localparam integer MAX_ERRORS_SHOWN = 10;        // dopo questi errori si smette di stamparli

    //=================================================================
    // Dimensioni dello stimolo, dei pesi e dei campioni attesi
    //=================================================================
    localparam integer NUM_SEQUENCES        = 2;           // sequenze nello stimolo (A, B)
    localparam integer SAMPLES_PER_SEQUENCE = L * H;       // 1024 campioni per sequenza
    localparam integer BEATS_PER_LAYER      = 14977;       // parole di pesi per layer (NBEAT in s4d_wload_axis.v)

    localparam integer NUM_STIM_SAMPLES = NUM_SEQUENCES * SAMPLES_PER_SEQUENCE;   // 2048 campioni in ingresso
    localparam integer NUM_EXPECTED     = NUM_SEQUENCES * SAMPLES_PER_SEQUENCE;   // 2048 campioni in uscita
    localparam integer NUM_WEIGHT_BEATS = NLAYER * BEATS_PER_LAYER;               // 89862 parole di pesi per sequenza

    //=================================================================
    // Memorie del testbench, riempite dai file .mem
    //=================================================================
    reg signed [W_ACT-1:0] stim_samples     [0:NUM_STIM_SAMPLES-1];   // campioni da mandare su s_axis_enc
    reg [AXI_DW-1:0]       weight_beats     [0:NUM_WEIGHT_BEATS-1];   // parole da mandare su s_axis_w
    reg signed [W_ACT-1:0] expected_samples [0:NUM_EXPECTED-1];       // campioni attesi su m_axis_out

    initial begin
        $readmemh(STIM_FILE,     stim_samples);
        $readmemh(WEIGHTS_FILE,  weight_beats);
        $readmemh(EXPECTED_FILE, expected_samples);

        // se un file manca o e' troppo corto, l'ultima cella resta a X:
        // meglio fermarsi subito che confrontare con dati non validi
        if (^stim_samples[NUM_STIM_SAMPLES-1] === 1'bx) begin
            $display("FAIL: %s non trovato o con meno di %0d campioni", STIM_FILE, NUM_STIM_SAMPLES);
            $finish;
        end
        if (^weight_beats[NUM_WEIGHT_BEATS-1] === 1'bx) begin
            $display("FAIL: %s non trovato o con meno di %0d parole", WEIGHTS_FILE, NUM_WEIGHT_BEATS);
            $finish;
        end
        if (^expected_samples[NUM_EXPECTED-1] === 1'bx) begin
            $display("FAIL: %s non trovato o con meno di %0d campioni", EXPECTED_FILE, NUM_EXPECTED);
            $finish;
        end
        $display("caricati %0d campioni di stimolo, %0d parole di pesi e %0d campioni attesi",
                 NUM_STIM_SAMPLES, NUM_WEIGHT_BEATS, NUM_EXPECTED);
    end

    //=================================================================
    // Segnali collegati al modulo
    //   reg  = pilotati dal testbench (ingressi del modulo)
    //   wire = pilotati dal modulo    (uscite del modulo)
    // I reg partono gia' con un valore, cosi' al tempo zero non ci sono X.
    //=================================================================
    reg                     clk   = 1'b0;                // clock
    reg                     rst_n = 1'b0;                // reset attivo basso (parte attivo)

    reg                     s_axis_enc_tvalid = 1'b0;    // il testbench ha un campione pronto
    wire                    s_axis_enc_tready;           // il core puo' accettarlo (solo mentre riceve la sequenza)
    reg  signed [W_ACT-1:0] s_axis_enc_tdata  = 16'd0;   // campione: un canale di un passo temporale
    reg                     s_axis_enc_tlast  = 1'b0;    // ultimo campione della sequenza

    reg                     s_axis_w_tvalid = 1'b0;      // il testbench ha una parola di pesi pronta
    wire                    s_axis_w_tready;             // il core la vuole (solo mentre carica un layer)
    reg  [AXI_DW-1:0]       s_axis_w_tdata  = 64'd0;     // parola di pesi
    reg                     s_axis_w_tlast  = 1'b0;      // non usato dal core: resta a 0

    wire                    m_axis_out_tvalid;           // il core ha un campione pronto
    reg                     m_axis_out_tready = 1'b1;    // il testbench puo' accettarlo
    wire signed [W_ACT-1:0] m_axis_out_tdata;            // campione in uscita
    wire                    m_axis_out_tlast;            // ultimo campione della sequenza

    wire [4:0]              status_err;                  // segnalazioni di errore del core (restano alzate)

    //=================================================================
    // Variabili del testbench
    //=================================================================
    integer stim_index;                    // indice del campione di stimolo che si sta mandando (0..2047)
    integer weight_index         = 0;      // indice della parola di pesi sul bus (0..89861, poi riparte)
    integer received_count       = 0;      // campioni ricevuti su m_axis_out nella fase in corso (devono essere 2048)
    integer error_count          = 0;      // errori trovati in tutta la simulazione

    integer cycle_count          = 0;      // cicli di clock dall'inizio della simulazione
    integer phase_start_cycle    = 0;      // ciclo in cui e' iniziata la fase in corso

    integer input_pause_percent  = 0;      // probabilita' di pausa sui campioni nella fase in corso
    integer weight_pause_percent = 0;      // probabilita' di pausa sui pesi nella fase in corso
    integer output_stall_percent = 0;      // probabilita' di stallo in uscita nella fase in corso
    integer input_seed           = RANDOM_SEED;          // stato del generatore casuale delle pause sui campioni
    integer weight_seed          = RANDOM_SEED + 500;    // stato del generatore casuale delle pause sui pesi
    integer output_seed          = RANDOM_SEED + 1000;   // stato del generatore casuale degli stalli

    reg     weight_beat_accepted = 1'b0;   // 1 se sull'ultimo fronte di salita il core ha preso la parola di pesi

    //=================================================================
    // Clock: si inverte ogni mezzo periodo
    //=================================================================
    always #(CLOCK_PERIOD_NS / 2) clk = ~clk;

    //=================================================================
    // Modulo sotto test
    //   Gli altri parametri restano al default. Le tabelle GELU e
    //   sigmoide sono lette dal modulo con i nomi gelu_lut.mem e
    //   sigmoid_lut.mem.
    //=================================================================
    s4d_core_axis #(
        .L      (L),
        .LOG2_L (LOG2_L),
        .F_SSM  (F_SSM)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .s_axis_enc_tvalid (s_axis_enc_tvalid),
        .s_axis_enc_tready (s_axis_enc_tready),
        .s_axis_enc_tdata  (s_axis_enc_tdata),
        .s_axis_enc_tlast  (s_axis_enc_tlast),
        .s_axis_w_tvalid   (s_axis_w_tvalid),
        .s_axis_w_tready   (s_axis_w_tready),
        .s_axis_w_tdata    (s_axis_w_tdata),
        .s_axis_w_tlast    (s_axis_w_tlast),
        .m_axis_out_tvalid (m_axis_out_tvalid),
        .m_axis_out_tready (m_axis_out_tready),
        .m_axis_out_tdata  (m_axis_out_tdata),
        .m_axis_out_tlast  (m_axis_out_tlast),
        .status_err        (status_err)
    );

    //=================================================================
    // Invio dei pesi su s_axis_w
    //   Il testbench tiene sempre pronta la prossima parola, come fa il
    //   DMA in scheda: e' il core a decidere quando prenderla, alzando
    //   tready mentre carica un layer. Dopo le 89862 parole dei 6 layer
    //   si riparte dalla prima, per la sequenza successiva.
    //   Si lavora in due tempi:
    //     fronte di salita   si annota se la parola e' stata accettata
    //     fronte di discesa  si prepara la parola per il ciclo successivo
    //=================================================================
    always @(posedge clk)
        weight_beat_accepted = rst_n && s_axis_w_tvalid && s_axis_w_tready;

    always @(negedge clk) begin
        if (!rst_n) begin
            // durante il reset si riparte dalla prima parola
            weight_index    = 0;
            s_axis_w_tvalid = 1'b0;
        end else begin
            // parola accettata: si passa alla successiva (dopo l'ultima si torna alla prima)
            if (weight_beat_accepted)
                weight_index = (weight_index == NUM_WEIGHT_BEATS - 1) ? 0 : weight_index + 1;

            // tvalid si puo' cambiare solo se sul bus non c'e' una parola ancora in attesa:
            // una volta alzato, deve restare a 1 finche' il core non la prende
            if (weight_beat_accepted || !s_axis_w_tvalid)
                s_axis_w_tvalid = (({$random(weight_seed)} % 100) >= weight_pause_percent);

            s_axis_w_tdata = weight_beats[weight_index];
        end
    end

    //=================================================================
    // Stalli in uscita: a ogni meta' ciclo si decide a caso se nel ciclo
    // successivo il testbench accetta il campione (tready = 1) oppure no.
    //   {$random(seed)} % 100  ->  numero casuale da 0 a 99
    //   (le graffe rendono il numero senza segno, cosi' il resto non e' mai negativo)
    // Con output_stall_percent = 0 il confronto e' sempre vero: tready resta a 1.
    //=================================================================
    always @(negedge clk)
        m_axis_out_tready = (({$random(output_seed)} % 100) >= output_stall_percent);

    //=================================================================
    // Ricezione su m_axis_out e confronto con i campioni attesi
    //   Un campione passa quando, su un fronte di salita, tvalid e tready
    //   sono entrambi a 1. Il campione numero received_count deve essere
    //   uguale a expected_samples[received_count].
    //   Questo blocco gira per conto suo, in parallelo alla sequenza
    //   principale che manda lo stimolo.
    //=================================================================
    always @(posedge clk) begin
        cycle_count = cycle_count + 1;   // contatore dei cicli, serve per misurare la durata

        if (!rst_n) begin
            // durante il reset si riparte a contare da zero (inizio di una nuova fase)
            received_count = 0;

        end else if (m_axis_out_tvalid && m_axis_out_tready) begin

            if (received_count >= NUM_EXPECTED) begin
                // il core ha emesso piu' campioni del previsto
                error_count = error_count + 1;
                if (error_count <= MAX_ERRORS_SHOWN)
                    $display("ERRORE: campione in piu' (n. %0d) = %h", received_count, m_axis_out_tdata);

            end else begin
                // confronto del campione:
                //   sequenza = received_count / 1024
                //   passo    = (received_count % 1024) / 128,  canale = received_count % 128
                // (!== riconosce anche X: un campione indefinito conta come errore)
                if (m_axis_out_tdata !== expected_samples[received_count]) begin
                    error_count = error_count + 1;
                    if (error_count <= MAX_ERRORS_SHOWN)
                        $display("ERRORE: sequenza %0d passo %0d canale %0d: ricevuto %h, atteso %h",
                                 received_count / SAMPLES_PER_SEQUENCE,
                                 (received_count % SAMPLES_PER_SEQUENCE) / H, received_count % H,
                                 m_axis_out_tdata, expected_samples[received_count]);
                end

                // tlast deve essere a 1 solo sull'ultimo campione di ogni sequenza
                if (m_axis_out_tlast !== ((received_count % SAMPLES_PER_SEQUENCE) == SAMPLES_PER_SEQUENCE - 1)) begin
                    error_count = error_count + 1;
                    if (error_count <= MAX_ERRORS_SHOWN)
                        $display("ERRORE: sequenza %0d passo %0d canale %0d: tlast = %b",
                                 received_count / SAMPLES_PER_SEQUENCE,
                                 (received_count % SAMPLES_PER_SEQUENCE) / H, received_count % H,
                                 m_axis_out_tlast);
                end
            end

            received_count = received_count + 1;
        end
    end

    //=================================================================
    // Reset: attivo per RESET_CYCLES cicli, ingressi a riposo.
    // Viene applicato e rilasciato a meta' ciclo, lontano dal fronte di salita.
    //=================================================================
    task apply_reset;
        begin
            @(negedge clk);
            rst_n             = 1'b0;
            s_axis_enc_tvalid = 1'b0;
            s_axis_enc_tlast  = 1'b0;
            s_axis_enc_tdata  = 16'd0;

            repeat (RESET_CYCLES) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
        end
    endtask

    //=================================================================
    // Invio di un campione su s_axis_enc
    //   Regola AXI-Stream: un campione passa quando, su un fronte di
    //   salita, tvalid e tready sono entrambi a 1. Finche' tready e' 0
    //   il campione deve restare fermo sul bus: succede per tutto il
    //   tempo in cui il core sta elaborando la sequenza precedente.
    //=================================================================
    task send_sample;
        input signed [W_ACT-1:0] sample_data;      // campione da mandare
        input                    sample_is_last;   // 1 solo per l'ultimo campione di una sequenza
        begin
            @(negedge clk);

            // pausa casuale PRIMA del campione: tvalid resta a 0 per uno o piu' cicli.
            // Si puo' fare solo qui, tra un campione e l'altro: una volta alzato,
            // tvalid non puo' piu' tornare a 0 finche' il campione non e' stato accettato.
            while (({$random(input_seed)} % 100) < input_pause_percent) begin
                s_axis_enc_tvalid = 1'b0;
                s_axis_enc_tlast  = 1'b0;
                @(negedge clk);
            end

            // il campione viene messo sul bus a meta' ciclo (fronte di discesa)
            s_axis_enc_tdata  = sample_data;
            s_axis_enc_tlast  = sample_is_last;
            s_axis_enc_tvalid = 1'b1;

            // se il core non e' pronto si aspetta, controllando a ogni meta' ciclo
            // (tready cambia solo sui fronti di salita, quindi a meta' ciclo e' stabile)
            while (!s_axis_enc_tready) @(negedge clk);

            // su questo fronte di salita il core accetta il campione
            @(posedge clk);
        end
    endtask

    //=================================================================
    // Una fase completa: reset, invio di tutto lo stimolo, attesa di
    // tutti i campioni in uscita, controlli finali della fase.
    //=================================================================
    task run_phase;
        input integer phase_number;            // numero della fase (solo per i messaggi)
        input integer enc_pause_percent;       // probabilita' di pausa sui campioni (0 = mai)
        input integer w_pause_percent;         // probabilita' di pausa sui pesi (0 = mai)
        input integer stall_percent;           // probabilita' di stallo in uscita (0 = mai)
        begin
            input_pause_percent  = enc_pause_percent;
            weight_pause_percent = w_pause_percent;
            output_stall_percent = stall_percent;
            $display("fase %0d: pause sui campioni %0d%%, pause sui pesi %0d%%, stalli in uscita %0d%%",
                     phase_number, enc_pause_percent, w_pause_percent, stall_percent);

            apply_reset;
            phase_start_cycle = cycle_count;

            // stato dopo il reset: niente in uscita, nessuna segnalazione di errore
            @(negedge clk);
            if (m_axis_out_tvalid !== 1'b0 || status_err !== 5'b00000) begin
                error_count = error_count + 1;
                $display("ERRORE: fase %0d, stato dopo il reset: tvalid=%b status_err=%b",
                         phase_number, m_axis_out_tvalid, status_err);
            end

            // invio dei 2048 campioni uno dopo l'altro;
            // tlast sull'ultimo campione di ogni sequenza (indici 1023 e 2047).
            // Il primo campione della seconda sequenza resta in attesa sul bus
            // finche' il core non ha finito di emettere la prima.
            for (stim_index = 0; stim_index < NUM_STIM_SAMPLES; stim_index = stim_index + 1)
                send_sample(stim_samples[stim_index],
                            (stim_index % SAMPLES_PER_SEQUENCE) == SAMPLES_PER_SEQUENCE - 1);

            // stimolo finito: si toglie tvalid dal bus
            @(negedge clk);
            s_axis_enc_tvalid = 1'b0;
            s_axis_enc_tlast  = 1'b0;

            // si aspetta che siano usciti tutti i 2048 campioni
            // (se non escono mai, interviene il tempo massimo piu' sotto)
            wait (received_count == NUM_EXPECTED);
            $display("fase %0d: ultimo campione ricevuto dopo %0d cicli di clock",
                     phase_number, cycle_count - phase_start_cycle);

            // altri cicli di attesa: se il core emette campioni in piu', vengono contati come errori
            repeat (CYCLES_AFTER_END) @(posedge clk);

            // segnalazioni di errore del core a fine fase
            //   bit 0 uscita sovrascritta, bit 1 lunghezza della sequenza errata,
            //   bit 2 scrittura che sorpassa la lettura, bit 3 saturazione nei filtri,
            //   bit 4 ready rotto tra filtri e mixing
            if (status_err !== EXPECTED_STATUS) begin
                error_count = error_count + 1;
                $display("ERRORE: fase %0d, status_err = %b (atteso %b)",
                         phase_number, status_err, EXPECTED_STATUS);
            end

            $display("fase %0d finita al tempo %0t: ricevuti %0d campioni, errori totali finora %0d",
                     phase_number, $time, received_count, error_count);
        end
    endtask

    //=================================================================
    // Sequenza principale
    //=================================================================
    initial begin
        // fase 1: massima velocita'
        run_phase(1, 0, 0, 0);

        // fase 2: pause e stalli casuali (prova che il core non perde e non duplica niente)
        run_phase(2, INPUT_PAUSE_PERCENT, WEIGHT_PAUSE_PERCENT, OUTPUT_STALL_PERCENT);

        // verdetto finale
        if (error_count == 0)
            $display("PASS: %0d campioni corretti (%0d sequenze) in ognuna delle 2 fasi",
                     NUM_EXPECTED, NUM_SEQUENCES);
        else
            $display("FAIL: %0d errori", error_count);
        $finish;
    end

    //=================================================================
    // Tempo massimo: se la simulazione dura troppo (per esempio perche'
    // il core si e' bloccato e non emette piu' campioni) si ferma con FAIL
    // invece di girare all'infinito.
    //=================================================================
    initial begin
        #(TIMEOUT_CYCLES * CLOCK_PERIOD_NS);
        $display("FAIL: tempo massimo superato, ricevuti %0d campioni su %0d nella fase in corso",
                 received_count, NUM_EXPECTED);
        $finish;
    end

endmodule

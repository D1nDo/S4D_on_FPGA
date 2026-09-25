`timescale 1ns / 1ps

//=====================================================================
// tb_s4d_encoder_ecg_axis -- testbench di s4d_encoder_ecg_axis
//
//   Manda al modulo lo stimolo di stim_10x12.mem (10 passi temporali da
//   6 parole a 32 bit) e confronta ogni dato in uscita con
//   expected_10x128.mem, prodotto da encoder_model.py.
//   Pesi e bias: enc_w.mem / enc_b.mem, caricati dal modulo stesso.
//
//   Lo stesso stimolo viene mandato DUE volte, con un reset in mezzo:
//     fase 1  alla massima velocita': nessuna pausa in ingresso,
//             uscita sempre accettata
//     fase 2  con pause casuali in ingresso (tvalid a 0) e stalli
//             casuali in uscita (tready a 0)
//   I dati attesi sono gli stessi nelle due fasi: pause e stalli possono
//   cambiare QUANDO esce un dato, mai QUALE dato esce.
//
//   Il testbench si giudica da solo: l'ultima riga stampata e'
//   "PASS" oppure "FAIL" con il numero di errori.
//
//   File necessari nella cartella da cui gira il simulatore:
//     stim_10x12.mem  expected_10x128.mem  enc_w.mem  enc_b.mem
//   Ordine per rigenerarli:
//     gen_encoder_stimulus.py, weight_gen.py, bias_gen.py, encoder_model.py
//
//   Sorgente del modulo (cartella rtl/ del repository):
//     s4d_encoder_ecg_axis.v
//=====================================================================

module tb_s4d_encoder_ecg_axis;

    //=================================================================
    // Parametri che si possono cambiare senza toccare il resto
    //=================================================================
    parameter         STIM_FILE            = "stim_10x12.mem";       // parole da mandare
    parameter         EXPECTED_FILE        = "expected_10x128.mem";  // dati attesi in uscita
    parameter integer RANDOM_SEED          = 1;     // seme dei numeri casuali della fase 2
    parameter integer INPUT_PAUSE_PERCENT  = 30;    // fase 2: probabilita' (0..99) di un ciclo di pausa in ingresso
    parameter integer OUTPUT_STALL_PERCENT = 30;    // fase 2: probabilita' (0..99) di un ciclo di stallo in uscita
    parameter         EXPECTED_SAT_STICKY  = 1'b0;  // valore atteso di sat_sticky (lo stampa encoder_model.py)

    //=================================================================
    // Costanti del testbench
    //=================================================================
    localparam integer CLOCK_PERIOD_NS  = 10;      // periodo del clock: 10 ns = 100 MHz
    localparam integer RESET_CYCLES     = 5;       // cicli di clock in cui il reset resta attivo
    localparam integer W_ACT            = 16;      // bit di un dato in uscita (come nel modulo)
    localparam integer CYCLES_AFTER_END = 300;     // cicli di attesa dopo l'ultimo dato, per scoprire dati in piu'
    localparam integer TIMEOUT_CYCLES   = 50000;   // durata massima di tutta la simulazione
    localparam integer MAX_ERRORS_SHOWN = 10;      // dopo questi errori si smette di stamparli

    //=================================================================
    // Dimensioni dello stimolo e dei valori attesi
    //=================================================================
    localparam integer NUM_TIMESTEPS      = 10;    // passi temporali nello stimolo
    localparam integer WORDS_PER_TIMESTEP = 6;     // parole da 32 bit per passo (12 campioni / 2)
    localparam integer H                  = 128;   // canali in uscita per passo

    localparam integer NUM_STIM_WORDS = NUM_TIMESTEPS * WORDS_PER_TIMESTEP;   // 60 parole in ingresso
    localparam integer NUM_EXPECTED   = NUM_TIMESTEPS * H;                    // 1280 dati in uscita

    //=================================================================
    // Memorie del testbench, riempite dai file .mem
    //=================================================================
    reg [31:0]             stim_words      [0:NUM_STIM_WORDS-1];   // parole da mandare su s_axis
    reg signed [W_ACT-1:0] expected_values [0:NUM_EXPECTED-1];     // dati attesi su m_axis

    initial begin
        $readmemh(STIM_FILE,     stim_words);
        $readmemh(EXPECTED_FILE, expected_values);

        // se un file manca o e' troppo corto, l'ultima cella resta a X:
        // meglio fermarsi subito che confrontare con dati non validi
        if (^stim_words[NUM_STIM_WORDS-1] === 1'bx) begin
            $display("FAIL: %s non trovato o con meno di %0d parole", STIM_FILE, NUM_STIM_WORDS);
            $finish;
        end
        if (^expected_values[NUM_EXPECTED-1] === 1'bx) begin
            $display("FAIL: %s non trovato o con meno di %0d valori", EXPECTED_FILE, NUM_EXPECTED);
            $finish;
        end
        $display("caricati %0d parole di stimolo e %0d valori attesi", NUM_STIM_WORDS, NUM_EXPECTED);
    end

    //=================================================================
    // Segnali collegati al modulo
    //   reg  = pilotati dal testbench (ingressi del modulo)
    //   wire = pilotati dal modulo    (uscite del modulo)
    // I reg partono gia' con un valore, cosi' al tempo zero non ci sono X.
    //=================================================================
    reg                     aclk          = 1'b0;    // clock
    reg                     aresetn       = 1'b0;    // reset attivo basso (parte attivo)

    reg                     s_axis_tvalid = 1'b0;    // il testbench ha una parola pronta
    wire                    s_axis_tready;           // il modulo puo' accettarla
    reg  [31:0]             s_axis_tdata  = 32'd0;   // parola: 2 campioni da 16 bit
    reg                     s_axis_tlast  = 1'b0;    // ultima parola dell'ultimo passo

    wire                    m_axis_tvalid;           // il modulo ha un dato pronto
    reg                     m_axis_tready = 1'b1;    // il testbench puo' accettarlo
    wire signed [W_ACT-1:0] m_axis_tdata;            // dato in uscita (un canale)
    wire                    m_axis_tlast;            // ultimo canale dell'ultimo passo

    wire                    sat_sticky;              // resta a 1 se un'uscita ha saturato
    wire                    err_frame;               // resta a 1 se tlast arriva fuori posto

    //=================================================================
    // Variabili del testbench
    //=================================================================
    integer stim_index;                    // indice della parola di stimolo che si sta mandando (0..59)
    integer received_count       = 0;      // dati ricevuti su m_axis nella fase in corso (devono essere 1280)
    integer error_count          = 0;      // errori trovati in tutta la simulazione

    integer cycle_count          = 0;      // cicli di clock dall'inizio della simulazione
    integer first_output_cycle   = 0;      // ciclo in cui e' uscito il primo dato della fase
    integer last_output_cycle    = 0;      // ciclo in cui e' uscito l'ultimo dato atteso della fase

    integer input_pause_percent  = 0;      // probabilita' di pausa in ingresso nella fase in corso
    integer output_stall_percent = 0;      // probabilita' di stallo in uscita nella fase in corso
    integer input_seed           = RANDOM_SEED;          // stato del generatore casuale delle pause
    integer output_seed          = RANDOM_SEED + 1000;   // stato del generatore casuale degli stalli

    //=================================================================
    // Clock: si inverte ogni mezzo periodo
    //=================================================================
    always #(CLOCK_PERIOD_NS / 2) aclk = ~aclk;

    //=================================================================
    // Modulo sotto test, con i parametri di default
    // (F_WENC = 31, file enc_w.mem ed enc_b.mem)
    //=================================================================
    s4d_encoder_ecg_axis dut (
        .aclk          (aclk),
        .aresetn       (aresetn),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tlast  (s_axis_tlast),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tlast  (m_axis_tlast),
        .sat_sticky    (sat_sticky),
        .err_frame     (err_frame)
    );

    //=================================================================
    // Stalli in uscita: a ogni meta' ciclo si decide a caso se nel ciclo
    // successivo il testbench accetta il dato (tready = 1) oppure no.
    //   {$random(seed)} % 100  ->  numero casuale da 0 a 99
    //   (le graffe rendono il numero senza segno, cosi' il resto non e' mai negativo)
    // Con output_stall_percent = 0 il confronto e' sempre vero: tready resta a 1.
    //=================================================================
    always @(negedge aclk)
        m_axis_tready = (({$random(output_seed)} % 100) >= output_stall_percent);

    //=================================================================
    // Ricezione su m_axis e confronto con i valori attesi
    //   Un dato passa quando, su un fronte di salita, tvalid e tready
    //   sono entrambi a 1. Il dato numero received_count deve essere
    //   uguale a expected_values[received_count].
    //   Questo blocco gira per conto suo, in parallelo alla sequenza
    //   principale che manda lo stimolo.
    //=================================================================
    always @(posedge aclk) begin
        cycle_count = cycle_count + 1;   // contatore dei cicli, serve per misurare la durata

        if (!aresetn) begin
            // durante il reset si riparte a contare da zero (inizio di una nuova fase)
            received_count = 0;

        end else if (m_axis_tvalid && m_axis_tready) begin

            if (received_count >= NUM_EXPECTED) begin
                // il modulo ha emesso piu' dati del previsto
                error_count = error_count + 1;
                if (error_count <= MAX_ERRORS_SHOWN)
                    $display("ERRORE: dato in piu' (n. %0d) = %h", received_count, m_axis_tdata);

            end else begin
                // si annota quando escono il primo e l'ultimo dato (per il controllo di durata)
                if (received_count == 0) first_output_cycle = cycle_count;
                last_output_cycle = cycle_count;

                // confronto del dato: passo = received_count / 128, canale = received_count % 128
                // (!== riconosce anche X: un dato indefinito conta come errore)
                if (m_axis_tdata !== expected_values[received_count]) begin
                    error_count = error_count + 1;
                    if (error_count <= MAX_ERRORS_SHOWN)
                        $display("ERRORE: passo %0d canale %0d: ricevuto %h, atteso %h",
                                 received_count / H, received_count % H,
                                 m_axis_tdata, expected_values[received_count]);
                end

                // tlast deve essere a 1 solo sull'ultimo dato dell'ultimo passo
                if (m_axis_tlast !== (received_count == NUM_EXPECTED - 1)) begin
                    error_count = error_count + 1;
                    if (error_count <= MAX_ERRORS_SHOWN)
                        $display("ERRORE: passo %0d canale %0d: tlast = %b",
                                 received_count / H, received_count % H, m_axis_tlast);
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
            @(negedge aclk);
            aresetn       = 1'b0;
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;
            s_axis_tdata  = 32'd0;

            repeat (RESET_CYCLES) @(posedge aclk);
            @(negedge aclk);
            aresetn = 1'b1;
        end
    endtask

    //=================================================================
    // Invio di una parola su s_axis
    //   Regola AXI-Stream: una parola passa quando, su un fronte di
    //   salita, tvalid e tready sono entrambi a 1. Finche' tready e' 0
    //   la parola deve restare ferma sul bus.
    //=================================================================
    task send_word;
        input [31:0] word_data;      // parola da mandare
        input        word_is_last;   // 1 solo per l'ultima parola dell'ultimo passo
        begin
            @(negedge aclk);

            // pausa casuale PRIMA della parola: tvalid resta a 0 per uno o piu' cicli.
            // Si puo' fare solo qui, tra una parola e l'altra: una volta alzato,
            // tvalid non puo' piu' tornare a 0 finche' la parola non e' stata accettata.
            while (({$random(input_seed)} % 100) < input_pause_percent) begin
                s_axis_tvalid = 1'b0;
                s_axis_tlast  = 1'b0;
                @(negedge aclk);
            end

            // la parola viene messa sul bus a meta' ciclo (fronte di discesa)
            s_axis_tdata  = word_data;
            s_axis_tlast  = word_is_last;
            s_axis_tvalid = 1'b1;

            // se il modulo non e' pronto si aspetta, controllando a ogni meta' ciclo
            // (tready cambia solo sui fronti di salita, quindi a meta' ciclo e' stabile)
            while (!s_axis_tready) @(negedge aclk);

            // su questo fronte di salita il modulo accetta la parola
            @(posedge aclk);
        end
    endtask

    //=================================================================
    // Una fase completa: reset, invio di tutto lo stimolo, attesa di
    // tutti i dati in uscita, controlli finali della fase.
    //=================================================================
    task run_phase;
        input integer phase_number;         // numero della fase (solo per i messaggi)
        input integer pause_percent;        // probabilita' di pausa in ingresso (0 = mai)
        input integer stall_percent;        // probabilita' di stallo in uscita (0 = mai)
        begin
            input_pause_percent  = pause_percent;
            output_stall_percent = stall_percent;
            $display("fase %0d: pause in ingresso %0d%%, stalli in uscita %0d%%",
                     phase_number, pause_percent, stall_percent);

            apply_reset;

            // stato dopo il reset: pronto a ricevere, niente in uscita, segnalazioni a 0
            @(negedge aclk);
            if (s_axis_tready !== 1'b1 || m_axis_tvalid !== 1'b0 ||
                sat_sticky    !== 1'b0 || err_frame     !== 1'b0) begin
                error_count = error_count + 1;
                $display("ERRORE: fase %0d, stato dopo il reset: tready=%b tvalid=%b sat_sticky=%b err_frame=%b",
                         phase_number, s_axis_tready, m_axis_tvalid, sat_sticky, err_frame);
            end

            // invio delle 60 parole una dopo l'altra; tlast solo sull'ultima
            for (stim_index = 0; stim_index < NUM_STIM_WORDS; stim_index = stim_index + 1)
                send_word(stim_words[stim_index], stim_index == NUM_STIM_WORDS - 1);

            // stimolo finito: si toglie tvalid dal bus
            @(negedge aclk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;

            // si aspetta che siano usciti tutti i 1280 dati
            // (se non escono mai, interviene il tempo massimo piu' sotto)
            wait (received_count == NUM_EXPECTED);

            // altri cicli di attesa: se il modulo emette dati in piu', vengono contati come errori
            repeat (CYCLES_AFTER_END) @(posedge aclk);

            // segnalazioni del modulo a fine fase
            if (err_frame !== 1'b0) begin
                error_count = error_count + 1;
                $display("ERRORE: fase %0d, err_frame = %b (atteso 0)", phase_number, err_frame);
            end
            if (sat_sticky !== EXPECTED_SAT_STICKY) begin
                error_count = error_count + 1;
                $display("ERRORE: fase %0d, sat_sticky = %b (atteso %b)",
                         phase_number, sat_sticky, EXPECTED_SAT_STICKY);
            end

            // durata: senza pause ne' stalli i 1280 dati devono uscire in 1280 cicli consecutivi,
            // cioe' un dato per ciclo e nessun ciclo vuoto tra un passo temporale e il successivo
            if (pause_percent == 0 && stall_percent == 0 &&
                (last_output_cycle - first_output_cycle) != NUM_EXPECTED - 1) begin
                error_count = error_count + 1;
                $display("ERRORE: fase %0d, dal primo all'ultimo dato %0d cicli (attesi %0d): ci sono cicli vuoti",
                         phase_number, last_output_cycle - first_output_cycle + 1, NUM_EXPECTED);
            end

            $display("fase %0d finita al tempo %0t: ricevuti %0d dati in %0d cicli, errori totali finora %0d",
                     phase_number, $time, received_count,
                     last_output_cycle - first_output_cycle + 1, error_count);
        end
    endtask

    //=================================================================
    // Sequenza principale
    //=================================================================
    initial begin
        // fase 1: massima velocita' (prova che tra un passo e l'altro non si perdono cicli)
        run_phase(1, 0, 0);

        // fase 2: pause e stalli casuali (prova che il modulo si ferma e riparte senza perdere dati)
        run_phase(2, INPUT_PAUSE_PERCENT, OUTPUT_STALL_PERCENT);

        // verdetto finale
        if (error_count == 0)
            $display("PASS: %0d dati corretti in ognuna delle 2 fasi", NUM_EXPECTED);
        else
            $display("FAIL: %0d errori", error_count);
        $finish;
    end

    //=================================================================
    // Tempo massimo: se la simulazione dura troppo (per esempio perche'
    // il modulo si e' bloccato e non emette piu' dati) si ferma con FAIL
    // invece di girare all'infinito.
    //=================================================================
    initial begin
        #(TIMEOUT_CYCLES * CLOCK_PERIOD_NS);
        $display("FAIL: tempo massimo superato, ricevuti %0d dati su %0d nella fase in corso",
                 received_count, NUM_EXPECTED);
        $finish;
    end

endmodule

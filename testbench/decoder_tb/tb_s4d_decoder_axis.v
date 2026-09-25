`timescale 1ns / 1ps

//=====================================================================
// tb_s4d_decoder_axis -- testbench di s4d_decoder_axis
//
//   Manda al modulo lo stimolo di stim_4x8x128.mem (4 sequenze da 8 passi
//   temporali x 128 canali) e confronta ogni parola in uscita con
//   expected_4x6.mem, prodotto da decoder_model.py: per ogni sequenza
//   5 logit a 32 bit e 1 parola di classe (0xC1 | flag di errore | classe).
//   Pesi e bias: l6_bn_w.mem, l6_bn_b.mem, cls_w.mem, cls_b.mem, caricati
//   dal modulo stesso.
//
//   PARAMETRI DEL MODULO IN QUESTA PROVA
//     K = 5, LOG2_K = 3, BN_SH = 13   valori del progetto ECG (block design)
//     L = 8, LOG2_L = 3               RIDOTTI: in scheda L = 1024, LOG2_L = 10
//   L e' ridotto di proposito. Con 8 passi lo stimolo resta piccolo e
//   leggibile e, soprattutto, un passo saltato o contato due volte sposta
//   la media di 1/8 e si vede subito; con 1024 passi la sposterebbe di
//   1/1024 e sparirebbe nell'arrotondamento. La logica provata e' la stessa:
//   L cambia solo la lunghezza dei contatori e lo shift della media.
//
//   Lo stesso stimolo viene mandato DUE volte, con un reset in mezzo:
//     fase 1  alla massima velocita': campioni uno per ciclo, sequenze
//             attaccate una all'altra, uscita sempre accettata
//     fase 2  con pause casuali in ingresso (tvalid a 0) e stalli casuali
//             in uscita (tready a 0)
//   Le parole attese sono le stesse nelle due fasi.
//
//   Il testbench si giudica da solo: l'ultima riga stampata e'
//   "PASS" oppure "FAIL" con il numero di errori.
//
//   File necessari nella cartella da cui gira il simulatore:
//     stim_4x8x128.mem  expected_4x6.mem
//     l6_bn_w.mem  l6_bn_b.mem  cls_w.mem  cls_b.mem
//   Ordine per rigenerarli:
//     weight_gen.py, bias_gen.py, gen_decoder_stimulus.py, decoder_model.py
//
//   Sorgenti del modulo (cartella rtl/ del repository):
//     s4d_decoder_axis.v  s4d_decoder.v  s4d_affine.v  s4d_meanpool.v  s4d_classifier.v
//=====================================================================

module tb_s4d_decoder_axis;

    //=================================================================
    // Parametri che si possono cambiare senza toccare il resto
    //=================================================================
    parameter         STIM_FILE            = "stim_4x8x128.mem";   // campioni da mandare
    parameter         EXPECTED_FILE        = "expected_4x6.mem";   // parole attese in uscita
    parameter integer RANDOM_SEED          = 1;     // seme dei numeri casuali della fase 2
    parameter integer INPUT_PAUSE_PERCENT  = 30;    // fase 2: probabilita' (0..99) di un ciclo di pausa in ingresso
    parameter integer OUTPUT_STALL_PERCENT = 30;    // fase 2: probabilita' (0..99) di un ciclo di stallo in uscita

    //=================================================================
    // Parametri con cui viene istanziato il modulo
    //=================================================================
    localparam integer H      = 128;   // canali per passo temporale
    localparam integer LOG2_H = 7;
    localparam integer L      = 8;     // passi temporali per sequenza (ridotto: in scheda 1024)
    localparam integer LOG2_L = 3;     // (ridotto: in scheda 10)
    localparam integer K      = 5;     // classi (progetto ECG)
    localparam integer LOG2_K = 3;
    localparam integer BN_SH  = 13;    // shift della normalizzazione finale (progetto ECG)
    localparam integer W_ACT  = 16;    // bit di un campione in ingresso

    //=================================================================
    // Costanti del testbench
    //=================================================================
    localparam integer CLOCK_PERIOD_NS  = 10;       // periodo del clock: 10 ns = 100 MHz
    localparam integer RESET_CYCLES     = 5;        // cicli di clock in cui il reset resta attivo
    localparam integer CYCLES_AFTER_END = 500;      // cicli di attesa dopo l'ultima parola, per scoprire parole in piu'
    localparam integer TIMEOUT_CYCLES   = 100000;   // durata massima di tutta la simulazione
    localparam integer MAX_ERRORS_SHOWN = 10;       // dopo questi errori si smette di stamparli

    //=================================================================
    // Dimensioni dello stimolo e delle parole attese
    //=================================================================
    localparam integer NUM_SEQUENCES        = 4;           // sequenze nello stimolo (A, B, C, D)
    localparam integer SAMPLES_PER_SEQUENCE = L * H;       // 1024 campioni per sequenza
    localparam integer WORDS_PER_PACKET     = K + 1;       // 5 logit + 1 parola di classe

    localparam integer NUM_STIM_SAMPLES = NUM_SEQUENCES * SAMPLES_PER_SEQUENCE;   // 4096 campioni in ingresso
    localparam integer NUM_EXPECTED     = NUM_SEQUENCES * WORDS_PER_PACKET;       // 24 parole in uscita

    //=================================================================
    // Memorie del testbench, riempite dai file .mem
    //=================================================================
    reg signed [W_ACT-1:0] stim_samples   [0:NUM_STIM_SAMPLES-1];   // campioni da mandare su s_axis
    reg [31:0]             expected_words [0:NUM_EXPECTED-1];       // parole attese su m_axis

    initial begin
        $readmemh(STIM_FILE,     stim_samples);
        $readmemh(EXPECTED_FILE, expected_words);

        // se un file manca o e' troppo corto, l'ultima cella resta a X:
        // meglio fermarsi subito che confrontare con dati non validi
        if (^stim_samples[NUM_STIM_SAMPLES-1] === 1'bx) begin
            $display("FAIL: %s non trovato o con meno di %0d campioni", STIM_FILE, NUM_STIM_SAMPLES);
            $finish;
        end
        if (^expected_words[NUM_EXPECTED-1] === 1'bx) begin
            $display("FAIL: %s non trovato o con meno di %0d parole", EXPECTED_FILE, NUM_EXPECTED);
            $finish;
        end
        $display("caricati %0d campioni di stimolo e %0d parole attese", NUM_STIM_SAMPLES, NUM_EXPECTED);
    end

    //=================================================================
    // Segnali collegati al modulo
    //   reg  = pilotati dal testbench (ingressi del modulo)
    //   wire = pilotati dal modulo    (uscite del modulo)
    // I reg partono gia' con un valore, cosi' al tempo zero non ci sono X.
    //=================================================================
    reg                     aclk          = 1'b0;    // clock
    reg                     aresetn       = 1'b0;    // reset attivo basso (parte attivo)

    reg                     s_axis_tvalid = 1'b0;    // il testbench ha un campione pronto
    wire                    s_axis_tready;           // il modulo puo' accettarlo (deve restare sempre a 1)
    reg  signed [W_ACT-1:0] s_axis_tdata  = 16'd0;   // campione: un canale di un passo temporale
    reg                     s_axis_tlast  = 1'b0;    // ultimo canale dell'ultimo passo della sequenza

    wire                    m_axis_tvalid;           // il modulo ha una parola pronta
    reg                     m_axis_tready = 1'b1;    // il testbench puo' accettarla
    wire [31:0]             m_axis_tdata;            // parola in uscita (logit oppure classe)
    wire                    m_axis_tlast;            // ultima parola del pacchetto (quella di classe)

    //=================================================================
    // Variabili del testbench
    //=================================================================
    integer stim_index;                    // indice del campione di stimolo che si sta mandando (0..4095)
    integer received_count       = 0;      // parole ricevute su m_axis nella fase in corso (devono essere 24)
    integer error_count          = 0;      // errori trovati in tutta la simulazione
    integer not_ready_cycles     = 0;      // cicli in cui s_axis_tready e' stato 0 nella fase in corso

    integer input_pause_percent  = 0;      // probabilita' di pausa in ingresso nella fase in corso
    integer output_stall_percent = 0;      // probabilita' di stallo in uscita nella fase in corso
    integer input_seed           = RANDOM_SEED;          // stato del generatore casuale delle pause
    integer output_seed          = RANDOM_SEED + 1000;   // stato del generatore casuale degli stalli

    //=================================================================
    // Clock: si inverte ogni mezzo periodo
    //=================================================================
    always #(CLOCK_PERIOD_NS / 2) aclk = ~aclk;

    //=================================================================
    // Modulo sotto test
    //   STREAM_LOGITS resta al default (1): prima i K logit, poi la classe.
    //   I nomi dei file .mem restano quelli di default del modulo.
    //=================================================================
    s4d_decoder_axis #(
        .H      (H),
        .LOG2_H (LOG2_H),
        .L      (L),
        .LOG2_L (LOG2_L),
        .K      (K),
        .LOG2_K (LOG2_K),
        .BN_SH  (BN_SH)
    ) dut (
        .aclk          (aclk),
        .aresetn       (aresetn),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tlast  (s_axis_tlast),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tlast  (m_axis_tlast)
    );

    //=================================================================
    // Stalli in uscita: a ogni meta' ciclo si decide a caso se nel ciclo
    // successivo il testbench accetta la parola (tready = 1) oppure no.
    //   {$random(seed)} % 100  ->  numero casuale da 0 a 99
    //   (le graffe rendono il numero senza segno, cosi' il resto non e' mai negativo)
    // Con output_stall_percent = 0 il confronto e' sempre vero: tready resta a 1.
    //=================================================================
    always @(negedge aclk)
        m_axis_tready = (({$random(output_seed)} % 100) >= output_stall_percent);

    //=================================================================
    // Ricezione su m_axis e confronto con le parole attese
    //   Una parola passa quando, su un fronte di salita, tvalid e tready
    //   sono entrambi a 1. La parola numero received_count deve essere
    //   uguale a expected_words[received_count].
    //   Questo blocco gira per conto suo, in parallelo alla sequenza
    //   principale che manda lo stimolo.
    //=================================================================
    always @(posedge aclk) begin
        if (!aresetn) begin
            // durante il reset si riparte a contare da zero (inizio di una nuova fase)
            received_count   = 0;
            not_ready_cycles = 0;

        end else begin
            // il decoder non puo' fermare chi gli manda i dati: tready deve restare a 1
            if (s_axis_tready !== 1'b1) not_ready_cycles = not_ready_cycles + 1;

            if (m_axis_tvalid && m_axis_tready) begin

                if (received_count >= NUM_EXPECTED) begin
                    // il modulo ha emesso piu' parole del previsto
                    error_count = error_count + 1;
                    if (error_count <= MAX_ERRORS_SHOWN)
                        $display("ERRORE: parola in piu' (n. %0d) = %h", received_count, m_axis_tdata);

                end else begin
                    // confronto della parola:
                    //   sequenza = received_count / 6, posizione nel pacchetto = received_count % 6
                    //   posizioni 0..4 = logit, posizione 5 = parola di classe
                    // (!== riconosce anche X: una parola indefinita conta come errore)
                    if (m_axis_tdata !== expected_words[received_count]) begin
                        error_count = error_count + 1;
                        if (error_count <= MAX_ERRORS_SHOWN)
                            $display("ERRORE: sequenza %0d parola %0d: ricevuto %h, atteso %h",
                                     received_count / WORDS_PER_PACKET, received_count % WORDS_PER_PACKET,
                                     m_axis_tdata, expected_words[received_count]);
                    end

                    // tlast deve essere a 1 solo sull'ultima parola di ogni pacchetto (quella di classe)
                    if (m_axis_tlast !== ((received_count % WORDS_PER_PACKET) == WORDS_PER_PACKET - 1)) begin
                        error_count = error_count + 1;
                        if (error_count <= MAX_ERRORS_SHOWN)
                            $display("ERRORE: sequenza %0d parola %0d: tlast = %b",
                                     received_count / WORDS_PER_PACKET, received_count % WORDS_PER_PACKET,
                                     m_axis_tlast);
                    end
                end

                received_count = received_count + 1;
            end
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
            s_axis_tdata  = 16'd0;

            repeat (RESET_CYCLES) @(posedge aclk);
            @(negedge aclk);
            aresetn = 1'b1;
        end
    endtask

    //=================================================================
    // Invio di un campione su s_axis
    //   Regola AXI-Stream: un campione passa quando, su un fronte di
    //   salita, tvalid e tready sono entrambi a 1.
    //=================================================================
    task send_sample;
        input signed [W_ACT-1:0] sample_data;      // campione da mandare
        input                    sample_is_last;   // 1 solo per l'ultimo campione di una sequenza
        begin
            @(negedge aclk);

            // pausa casuale PRIMA del campione: tvalid resta a 0 per uno o piu' cicli.
            // Si puo' fare solo qui, tra un campione e l'altro: una volta alzato,
            // tvalid non puo' piu' tornare a 0 finche' il campione non e' stato accettato.
            while (({$random(input_seed)} % 100) < input_pause_percent) begin
                s_axis_tvalid = 1'b0;
                s_axis_tlast  = 1'b0;
                @(negedge aclk);
            end

            // il campione viene messo sul bus a meta' ciclo (fronte di discesa)
            s_axis_tdata  = sample_data;
            s_axis_tlast  = sample_is_last;
            s_axis_tvalid = 1'b1;

            // se il modulo non fosse pronto si aspetterebbe qui (per il decoder non deve succedere mai)
            while (!s_axis_tready) @(negedge aclk);

            // su questo fronte di salita il modulo accetta il campione
            @(posedge aclk);
        end
    endtask

    //=================================================================
    // Una fase completa: reset, invio di tutto lo stimolo, attesa di
    // tutte le parole in uscita, controlli finali della fase.
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

            // stato dopo il reset: pronto a ricevere, niente in uscita
            @(negedge aclk);
            if (s_axis_tready !== 1'b1 || m_axis_tvalid !== 1'b0) begin
                error_count = error_count + 1;
                $display("ERRORE: fase %0d, stato dopo il reset: tready=%b tvalid=%b",
                         phase_number, s_axis_tready, m_axis_tvalid);
            end

            // invio dei 4096 campioni uno dopo l'altro;
            // tlast sull'ultimo campione di ogni sequenza (indici 1023, 2047, 3071, 4095)
            for (stim_index = 0; stim_index < NUM_STIM_SAMPLES; stim_index = stim_index + 1)
                send_sample(stim_samples[stim_index],
                            (stim_index % SAMPLES_PER_SEQUENCE) == SAMPLES_PER_SEQUENCE - 1);

            // stimolo finito: si toglie tvalid dal bus
            @(negedge aclk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;

            // si aspetta che siano uscite tutte le 24 parole
            // (se non escono mai, interviene il tempo massimo piu' sotto)
            wait (received_count == NUM_EXPECTED);

            // altri cicli di attesa: se il modulo emette parole in piu', vengono contate come errori
            repeat (CYCLES_AFTER_END) @(posedge aclk);

            // il decoder non deve mai aver abbassato tready in ingresso
            if (not_ready_cycles != 0) begin
                error_count = error_count + 1;
                $display("ERRORE: fase %0d, s_axis_tready a 0 per %0d cicli (deve restare sempre a 1)",
                         phase_number, not_ready_cycles);
            end

            $display("fase %0d finita al tempo %0t: ricevute %0d parole, errori totali finora %0d",
                     phase_number, $time, received_count, error_count);
        end
    endtask

    //=================================================================
    // Sequenza principale
    //=================================================================
    initial begin
        // fase 1: massima velocita', sequenze attaccate una all'altra
        run_phase(1, 0, 0);

        // fase 2: pause e stalli casuali (prova che il modulo non perde e non duplica niente)
        run_phase(2, INPUT_PAUSE_PERCENT, OUTPUT_STALL_PERCENT);

        // verdetto finale
        if (error_count == 0)
            $display("PASS: %0d parole corrette (%0d pacchetti) in ognuna delle 2 fasi",
                     NUM_EXPECTED, NUM_SEQUENCES);
        else
            $display("FAIL: %0d errori", error_count);
        $finish;
    end

    //=================================================================
    // Tempo massimo: se la simulazione dura troppo (per esempio perche'
    // il modulo si e' bloccato e non emette piu' parole) si ferma con FAIL
    // invece di girare all'infinito.
    //=================================================================
    initial begin
        #(TIMEOUT_CYCLES * CLOCK_PERIOD_NS);
        $display("FAIL: tempo massimo superato, ricevute %0d parole su %0d nella fase in corso",
                 received_count, NUM_EXPECTED);
        $finish;
    end

endmodule

@php
  $number_of_questions = 30;
  $questionsPerRow = 15;
  $answarePerRow = 30;
  $options = ['ক', 'খ', 'গ', 'ঘ'];
  $type ="2nd";
@endphp
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="theme-color" content="blue" />
  <title>OMR quiz exam</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>
      body {
          font-family: 'Tiro Bangla', sans-serif;
          margin: 0;
          padding: 0;
          background: #fdfdfd;
      }
      h5 {
          text-align: center;
          margin-bottom: 15px;
      }
      .container {
          /* width: 95%; */
          max-width: 1200px;
          margin: 20px auto;
      }
      table {
          width: 100%;
          border-collapse: collapse;
      }
      table, th, td {
          border: 1px solid #df96ac;
      }
      th, td {
          text-align: center;
          padding: 5px;
      }
      th {
          background-color: #f7c4df;
      }
      .circle-label {
          width: 25px;
          height: 25px;
          border: 1px solid #df96ac;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: bold;
          cursor: pointer;
          color: #df96ac;
          font-size: 14px;
      }
      .form-check-input {
          display: none;
      }
      .form-check-input:checked + .circle-label {
          background-color: #302a29;
          color: white;
          border-color: #aa697b;
      }
      .form-check {
          padding: 2px 0;
      }
      /* Layout system (replacement for Bootstrap rows/cols) */
      .row {
          display: flex;
          flex-wrap: wrap;
          gap: 15px;
      }
      .col-12 { flex: 1 1 100%; }
      .col-9 { flex: 0 0 75%; }
      .col-3 { flex: 0 0 22.9%; display: flex; flex-direction: column; gap: 10px; }
      
      .box {
          border: 1px solid #df96ac;
          border-radius: 5px;
          padding: 10px;
          background: #fff;
      }
      input[type="text"] {
          width: 90%;
          padding: 6px;
          font-size: 14px;
          border: 1px solid #ccc;
          border-radius: 4px;
          text-align: center;
      }
      #total-marks {
          font-size: 80px;
          font-weight: bold;
          color: #302a29;
      }
      .btn {
          display: block;
          width: 100%;
          padding: 12px;
          background: #302a29;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 16px;
          cursor: pointer;
          text-align: center;
          font-family: 'Tiro Bangla';
      }
      .btn:hover {
          background: #000;
      }
  </style>
  <link href='https://fonts.googleapis.com/css?family=Tiro Bangla' rel='stylesheet'>

    <style>
          body
        {
            width:100%;
            background: rgb(204,204,204);
            overflow-x: hidden;
        }
        page[size="A4"] {
            background: white;
            width: 21cm;
            height: 29.7cm;
            display: block;
            margin: 0 auto;
            margin-bottom: 0.5cm;
            box-shadow: 0 0 0.1cm rgba(0,0,0,0.5);
            -o-box-shadow: 0 0 0.1cm rgba(0,0,0,0.5);
            -webkit-box-shadow: 0 0 0.1cm rgba(0,0,0,0.5);
            -moz-box-shadow: 0 0 0.1cm rgba(0,0,0,0.5);
        }
        @page {
            size: A4;
            margin: 0;
        }
        @media print {
            body, page[size="A4"] {
                margin: 0;
                box-shadow: 0;
            }
        }
        .first
        {
            width:100%;
            height:1080px;
            margin:auto; padding:5px 0px 0px 0px;
        }
        .second
        {
            width:728px;
            height:1060px;
            margin:auto; background-color:red;
        }
        th,td{
            font-size:13px;	padding:3px;
        }
        #footer
        {
            width:400px;
            margin:auto;
            background-color:#FFF;
        }
        #footer a
        {
            text-decoration:none;
            text-align:center;
            font-size:12px;
        }
        @media print{
            #print{
                display:none;
            }
            .print{
                display:none;
            }
        }
        .upon{
            width:400px;
            height:auto;
            margin:auto;
        }
        .upon h3{font-size:22px}
        #site_header_logo{
            position: relative;
            width:105px;
            height:100px;
            float:left;
            margin-top:-10px;
            margin-left:10px;
        }
        a{
            text-decoration:none;
            color:inherit;
        }
        a.tooltip {outline:none; }
        a.tooltip strong {line-height:30px;}
        a.tooltip:hover {text-decoration:none;}
        a.tooltip span {
            z-index:10;display:none; padding:14px 20px;
            margin-top:30px; margin-left:2px;
            width:300px; line-height:16px;
        }
        a.tooltip:hover span{
            display:inline; position:absolute; color:#111;
            border:1px solid #DCA; background:#fffAF0;
            font-size:16px;
        }
        .callout {z-index:20;position:absolute;top:30px;border:0;left:-12px;}

        /*CSS3 extras*/
        a.tooltip span
        {
            border-radius:4px;
            box-shadow: 5px 5px 8px #CCC;
        }
        
        @media print {
  .box {
    border: 1px solid #df96ac !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
}
    </style>
    <link href='https://fonts.googleapis.com/css?family=Tiro Bangla' rel='stylesheet'>


</head>
<body>


    <br class="print">

<button type="button" class="btn-fill-lg btn-gradient-yellow btn-hover-bluedark print">
<a href="http://127.0.0.1:8000/result/report/mark-sheet">Back to marksheet</a>
</button>
<button class="print" style="margin-top: 10px;margin-right: 10px" onclick="window.print()">Print</button>
<br class="print" />
<br class="print" />

<page size="A4">
    <div class="first">


        <div style="width:90%; height:auto; margin:auto">
            
            <div id="site_header_logo">
                <br>
                <br>
                <a href="#" style="text-decoration:none; cursor:pointer">
                    <img src="http://127.0.0.1:8000/LogoImageFolder/1-3-1640952397-logo new.png" width="100" ></a>
            </div>


            <div class="upon">
                <h1 align="center">আল কুরআন একাডেমী</h1>
                <h5 align="center" style="margin-top:-10px">কধুরখিল, বোয়ালখালী, চট্টগ্রাম, বাংলাদেশ.</h5>
                <h5 align="center" style="margin-top:-10px">
                    ক্লাস: দাখিল - X, সেশন: 2022, সেমিস্টার: দ্বিতীয় টার্মিনাল, গ্রুপ: A
                </h5>
                <h5 align="center" style="margin-top:-10px">
                    বিষয়: বাংলা, মোট নম্বর: 200
                </h5>
            </div>
            <br>
            <hr>
        </div>
        
        <div style="clear:both; width:100%; height:10px"></div>



    <div class="container" style="padding-left:20px;padding-right:20px;">
      <div class="row box">
          <div class="col-9">
              <table>
                  @for ($rowStart = 1; $rowStart <= $number_of_questions; $rowStart += $questionsPerRow)
                      @php
                          $rowEnd = min($rowStart + $questionsPerRow - 1, $number_of_questions);
                      @endphp
                      <tr>
                          @for ($i = $rowStart; $i <= $rowEnd; $i++)
                              <th>
                                {{-- প্রশ্ন  --}}
                                {{ (new \Rakibhstu\Banglanumber\NumberToBangla)->bnNum($i) }}
                              </th>
                          @endfor
                      </tr>
                      <tr>
                          @for ($i = $rowStart; $i <= $rowEnd; $i++)
                              <td>
                                  <div style="display:flex;flex-direction:column;align-items:center;gap:3px;">
                                      @foreach ($options as $option)
                                          @php $optionId = "q{$i}{$option}"; @endphp
                                          <div class="form-check">
                                              <input class="form-check-input student-answer" type="radio" name="q{{ $i }}" id="{{ $optionId }}">
                                              <label class="circle-label" for="{{ $optionId }}">{{ $option }}</label>
                                          </div>
                                      @endforeach
                                  </div>
                              </td>
                          @endfor
                      </tr>
                  @endfor
              </table>
          </div>

          <div class="col-3">
              <table style="width:100%;height:100%;">
                  <tr><th>শিক্ষার্থী আইডি</th></tr>
                  <tr><td>
                    @if ($type == "1st")
                        <input type="text" name="student_id">
                    @else  
                        <div style="height:50px"></div> 
                    @endif
                  </td></tr>
              </table>

              <table style="width:100%;height:100%;">
                  <tr><th>বিষয় কোড</th></tr>
                  <tr><td>
                    @if ($type == "1st")
                        <input type="text" name="subject_code">
                    @else  
                        <div style="height:50px"></div> 
                    @endif
                  </td></tr>
              </table>

              <table style="width:100%;height:100%;">
                  <tr><th>মোট নম্বর</th></tr>
                  <tr>
                    <td>
                        @if ($type == "1st")
                           <div id="total-marks">০</div>
                        @else  
                           <div style="height:106px"></div> 
                        @endif
                    </td>
                  </tr>
              </table>

              @if ($type == "1st")
                  <button type="button" class="btn">উত্তর জমা দিন</button>
              @endif
              
          </div>
      </div>
  </div>


        <div style="padding-left:50px;padding-right:50px;">
            <h4>প্রশ্ন ১: বাংলাদেশের জাতীয় মাছের নাম কী?</h4>
            <div style="width:100%;display:flex;">
                <div style="width:25%;word-wrap: break-word;">
                   (ক) রুই
                </div>
                <div style="width:25%;word-wrap: break-word;">
                   (খ) ইলিশ
                </div>
                <div style="width:25%;word-wrap: break-word;">
                   (গ) কাতলা
                </div>
                <div style="width:25%;word-wrap: break-word;">
                   (ঘ) চিংড়ি
                </div>
            </div>
        </div> 

        <div style="padding-left:50px;padding-right:50px;">
            <h4>প্রশ্ন ২: কোন শহরকে "মসজিদের শহর" বলা হয়?</h4>
            <div style="width:100%;display:flex;">
                <div style="width:25%;word-wrap: break-word;">
                   (ক) ঢাকা
                </div>
                <div style="width:25%;word-wrap: break-word;">
                   (খ) চট্টগ্রাম
                </div>
                <div style="width:25%;word-wrap: break-word;">
                   (গ) সিলেট
                </div>
                <div style="width:25%;word-wrap: break-word;">
                   (ঘ) খুলনা
                </div>
            </div>
        </div> 



        <div id="footer">
            <center>
            </center>
        </div>
    </div>
</page>


  {{-- <div class="container">
      <h5>সঠিক উত্তর (শিক্ষক কর্তৃক নির্ধারিত)</h5>
      <div class="box">
          <table>
              @for ($rowStart = 1; $rowStart <= $number_of_questions; $rowStart += $answarePerRow)
                  @php
                      $rowEnd = min($rowStart + $answarePerRow - 1, $number_of_questions);
                  @endphp
                  <tr>
                      @for ($i = $rowStart; $i <= $rowEnd; $i++)
                          <th>উত্তর {{ (new \Rakibhstu\Banglanumber\NumberToBangla)->bnNum($i) }}</th>
                      @endfor
                  </tr>
                  <tr>
                      @for ($i = $rowStart; $i <= $rowEnd; $i++)
                          <td>
                              <div style="display:flex;flex-direction:column;align-items:center;gap:3px;">
                                  @foreach ($options as $option)
                                      @php $optionId = "a{$i}" . strtolower($option); @endphp
                                      <div class="form-check">
                                          <input class="form-check-input teacher-answer" type="radio" name="a{{ $i }}" id="{{ $optionId }}">
                                          <label class="circle-label" for="{{ $optionId }}">{{ $option }}</label>
                                      </div>
                                  @endforeach
                              </div>
                          </td>
                      @endfor
                  </tr>
              @endfor
          </table>
      </div>
  </div>

  <div class="container">
      <h5>শিক্ষার্থীর উত্তর</h5>
      <div class="row box">
          <div class="col-9">
              <table>
                  @for ($rowStart = 1; $rowStart <= $number_of_questions; $rowStart += $questionsPerRow)
                      @php
                          $rowEnd = min($rowStart + $questionsPerRow - 1, $number_of_questions);
                      @endphp
                      <tr>
                          @for ($i = $rowStart; $i <= $rowEnd; $i++)
                              <th>প্রশ্ন {{ (new \Rakibhstu\Banglanumber\NumberToBangla)->bnNum($i) }}</th>
                          @endfor
                      </tr>
                      <tr>
                          @for ($i = $rowStart; $i <= $rowEnd; $i++)
                              <td>
                                  <div style="display:flex;flex-direction:column;align-items:center;gap:3px;">
                                      @foreach ($options as $option)
                                          @php $optionId = "q{$i}{$option}"; @endphp
                                          <div class="form-check">
                                              <input class="form-check-input student-answer" type="radio" name="q{{ $i }}" id="{{ $optionId }}">
                                              <label class="circle-label" for="{{ $optionId }}">{{ $option }}</label>
                                          </div>
                                      @endforeach
                                  </div>
                              </td>
                          @endfor
                      </tr>
                  @endfor
              </table>
          </div>

          <div class="col-3">
              <table style="width:100%;height:100%;">
                  <tr><th>শিক্ষার্থী আইডি</th></tr>
                  <tr><td><input type="text" name="student_id"></td></tr>
              </table>

              <table style="width:100%;height:100%;">
                  <tr><th>বিষয় কোড</th></tr>
                  <tr><td><input type="text" name="subject_code"></td></tr>
              </table>

              <table style="width:100%;height:100%;">
                  <tr><th>মোট নম্বর</th></tr>
                  <tr><td><div id="total-marks">০</div></td></tr>
              </table>

              <button type="button" class="btn">উত্তর জমা দিন</button>
          </div>
      </div>
  </div> --}}

<script>
document.querySelectorAll('.student-answer').forEach(input => {
    input.addEventListener('change', calculateMarks);
});
document.querySelectorAll('.teacher-answer').forEach(input => {
    input.addEventListener('change', calculateMarks);
});
function calculateMarks() {
    let total = 0;
    for (let i = 1; i <= {{ $number_of_questions }}; i++) {
        const correct = document.querySelector(`input[name="a${i}"]:checked`);
        const student = document.querySelector(`input[name="q${i}"]:checked`);
        if (correct && student) {
            const correctLabel = document.querySelector(`label[for="${correct.id}"]`);
            const studentLabel = document.querySelector(`label[for="${student.id}"]`);
            if (correctLabel.textContent.trim() === studentLabel.textContent.trim()) {
                total++;
            }
        }
    }
    document.getElementById('total-marks').innerText = toBanglaDigit(total);
    function toBanglaDigit(number) {
        const engToBn = {'0':'০','1':'১','2':'২','3':'৩','4':'৪','5':'৫','6':'৬','7':'৭','8':'৮','9':'৯'};
        return number.toString().replace(/[0-9]/g, d => engToBn[d]);
    }
}
</script>
</body>
</html>

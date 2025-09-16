<?php
error_reporting(E_ALL);
ini_set('display_errors', 1);

if (isset($_POST['query'])) {
    $query = $_POST['query'];

    $flaskUrl = "http://127.0.0.1:5000/query";

    $postData = json_encode(['query' => $query]);


    $ch = curl_init($flaskUrl);
    curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "POST");
    curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json',
        'Content-Length: ' . strlen($postData)
    ]);


    $result = curl_exec($ch);
    if ($result === false) {
        $error = curl_error($ch);
        curl_close($ch);
        echo "cURL error: $error";
        exit;
    }
    curl_close($ch);

    $responseData = json_decode($result, true);
    if (isset($responseData['response'])) {

        $decodedResponse = html_entity_decode($responseData['response'], ENT_QUOTES, 'UTF-8');

        if (isset($responseData['is_code']) && $responseData['is_code'] === true) {

            echo "<pre><code class='language-plaintext'>"
               . htmlspecialchars($decodedResponse)
               . "</code></pre>";


            echo "<script>Prism.highlightAll();</script>";
        } else {

            echo "<pre>" . $decodedResponse . "</pre>";
        }
    } else {
        echo "<pre>No response received.</pre>";
    }
} else {
    echo "No query provided.";
}
?>

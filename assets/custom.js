var dagfuncs = window.dashAgGridFunctions = window.dashAgGridFunctions || {};

dagfuncs.addEdits = function(params) {
    console.log(params);  // Debugging line
    if (!params.data) {
        console.log('addEdits was called with params.data undefined');
    }
    if (params && params.data && params.data.changes && params.colDef && params.colDef.field) {
        var newList = JSON.parse(params.data.changes)
        newList.push(params.colDef.field)
        params.data.changes = JSON.stringify(newList)
    } else if (params && params.data && params.colDef && params.colDef.field) {
        params.data.changes = JSON.stringify([params.colDef.field])
    }
    if (params && params.data && params.colDef && params.colDef.field) {
        params.data[params.colDef.field] = params.newValue
    }
    return true;
}

dagfuncs.highlightEdits = function(params) {
    console.log(params);  // Debugging line
    if (!params.data) {
        console.log('highlightEdits was called with params.data undefined');
    }
    if (params && params.data && params.data.changes && params.colDef && params.colDef.field) {
        if (JSON.parse(params.data.changes).includes(params.colDef.field))
            {return true}
    }
    return false;
}
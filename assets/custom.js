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


var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

dagcomponentfuncs.BarGuage = function (params) {
    const value = params.value;
    const columnField = params.colDef.field;
    const rowData = params.api.getDisplayedRowAtIndex(params.rowIndex).data;
    const allValues = params.api.rowModel.rowsToDisplay.map(row => row.data[columnField]);

    const min = Math.min(...allValues);
    const max = Math.max(...allValues);

    var percentage = ((value - min) / (max - min)) * 100;
    var barColor = params.colDef.guageColor;

    return React.createElement(
        'div',
        {
            style: {
                width: '100%',
                height: '100%',
                backgroundColor: 'transparent',
                position: 'relative'
            }
        },
        React.createElement('div', {
            style: {
                height: '100%',
                width: `${percentage}%`,
                backgroundColor: barColor
            }
        }),
        React.createElement(
            'div',
            {
                style: {
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    textAlign: 'center'
                }
            },
            value
        )
    );
}


dagcomponentfuncs.HeatMap = function(params) {
    const value = params.value;
    const columnField = params.colDef.field;
    const allValues = params.api.rowModel.rowsToDisplay.map(row => row.data[columnField]);

    const min = Math.min(...allValues);
    const max = Math.max(...allValues);

    const percentage = ((value - min) / (max - min)) * 100;
    // Color gradient from sage green to deep red
    const red = Math.min(139, Math.floor(139 * (percentage / 100)));
    const green = Math.max(175, 175 - Math.floor(175 * (percentage / 100)));
    const blue = Math.max(136, 136 - Math.floor(136 * (percentage / 100)));
    const color = `rgb(${red}, ${green}, ${blue})`;

    return React.createElement(
        'div',
        {
            style: {
                width: '100%',
                height: '100%',
                backgroundColor: color,
                textAlign: 'center'
            }
        },
        value
    );
}